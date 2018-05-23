package org.broadinstitute.hellbender.tools.spark.sv.evidence;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.learner.ObjFunction;
import com.google.common.annotations.VisibleForTesting;
import htsjdk.samtools.CigarElement;
import htsjdk.samtools.CigarOperator;
import htsjdk.samtools.TextCigarCodec;
import htsjdk.tribble.AbstractFeatureReader;
import htsjdk.tribble.bed.BEDFeature;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.broadinstitute.hellbender.engine.FeatureDataSource;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.spark.sv.StructuralVariationDiscoveryArgumentCollection;
import org.broadinstitute.hellbender.tools.spark.sv.utils.*;
import org.broadinstitute.hellbender.tools.spark.utils.IntHistogram;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.gcs.BucketUtils;
import org.broadinstitute.hellbender.utils.io.IOUtils;

import java.io.*;
import java.util.*;

/**
 * A class that acts as a filter for BreakpointEvidence.
 * Features are calculated according to evidence type, overlap information, mapping quality, etc.
 * A trained classifier scores the probability the evidence overlaps a breakpoint interval, and passes evidence above
 * the specified threshold.
 */
public final class XGBoostEvidenceFilter implements Iterator<BreakpointEvidence> {
    private static final boolean USE_FAST_MATH_EXP = true;

    private static final List<String> DEFAULT_EVIDENCE_TYPE_ORDER = Arrays.asList(
            "TemplateSizeAnomaly", "MateUnmapped", "InterContigPair",
            "SplitRead", "LargeIndel", "WeirdTemplateSize", "SameStrandPair", "OutiesPair"
    );
    private static final Map<String, Integer> evidenceTypeMap = evidenceTypeOrderToImmutableMap(DEFAULT_EVIDENCE_TYPE_ORDER);
    private static final String DEFAULT_PREDICTOR_RESOURCE_PATH = "/large/sv_evidence_classifier.bin";

    private final PartitionCrossingChecker partitionCrossingChecker;

    private final Predictor predictor;
    private final double thresholdProbability;
    private final ReadMetadata readMetadata;

    private final EvidenceOverlapChecker evidenceOverlapChecker;
    private final Map<BreakpointEvidence, UnscaledOverlapInfo> rawFeatureCache;

    private Iterator<SVIntervalTree.Entry<List<BreakpointEvidence>>> treeItr;
    private Iterator<BreakpointEvidence> listItr;
    private final FeatureDataSource<BEDFeature> genomeGaps;
    private final FeatureDataSource<BEDFeature> umapS100Mappability;

    XGBoostEvidenceFilter(
            final Iterator<BreakpointEvidence> evidenceItr,
            final ReadMetadata readMetadata,
            final StructuralVariationDiscoveryArgumentCollection.FindBreakpointEvidenceSparkArgumentCollection params,
            final PartitionCrossingChecker partitionCrossingChecker
    ) {
        if(params.svGenomeGapsFile == null) {
            throw new IllegalArgumentException("XGBoostEvidenceFilter requires non-null sv-genome-gaps-file");
        }
        if(params.svGenomeUmapS100File == null) {
            throw new IllegalArgumentException("XGBoostEvidenceFilter requires non-null sv-genome-umap-s100-file");
        }
        predictor = loadPredictor(params.svEvidenceFilterModelFile);
        genomeGaps = new FeatureDataSource<>(params.svGenomeGapsFile);
        umapS100Mappability = new FeatureDataSource<>(params.svGenomeUmapS100File);

        this.partitionCrossingChecker = partitionCrossingChecker;
        thresholdProbability = params.svEvidenceFilterThresholdProbability;
        this.readMetadata = readMetadata;

        evidenceOverlapChecker = new EvidenceOverlapChecker(evidenceItr, readMetadata, params.minEvidenceMapQ);
        rawFeatureCache = new HashMap<>();

        listItr = null;
        treeItr = evidenceOverlapChecker.getTreeItr();
    }

    private static Map<String, Integer> evidenceTypeOrderToImmutableMap(final List<String> evidenceTypeOrder) {
        final HashMap<String, Integer> evidenceTypeMap = new HashMap<>();
        for(int index = 0; index < evidenceTypeOrder.size(); ++index) {
            evidenceTypeMap.put(evidenceTypeOrder.get(index), index);
        }
        return Collections.unmodifiableMap(evidenceTypeMap);
    }

    public static Predictor loadPredictor(final String modelFileLocation) {
        ObjFunction.useFastMathExp(USE_FAST_MATH_EXP);
        try(final InputStream inputStream = modelFileLocation == null ?
                resourcePathToInputStream(DEFAULT_PREDICTOR_RESOURCE_PATH) : BucketUtils.openFile(modelFileLocation)) {
            return new Predictor(inputStream);
        } catch(Exception e) {
            throw new GATKException(
                    "Unable to load predictor from classifier file " + (modelFileLocation == null ? DEFAULT_PREDICTOR_RESOURCE_PATH : modelFileLocation)
                            + ": " + e.getMessage()
            );
        }
    }

    private static InputStream resourcePathToInputStream(final String resourcePath) throws IOException {
        final InputStream inputStream = XGBoostEvidenceFilter.class.getResourceAsStream(resourcePath);
        return AbstractFeatureReader.hasBlockCompressedExtension(resourcePath) ?
                IOUtils.makeZippedInputStream(new BufferedInputStream(inputStream))
                : inputStream;
    }

    @Override
    public boolean hasNext() {
        if ( listItr != null && listItr.hasNext() ) {
            return true;
        }
        listItr = null;
        boolean result = false;
        while ( !result && treeItr.hasNext() ) {
            final SVIntervalTree.Entry<List<BreakpointEvidence>> entry = treeItr.next();
            final SVInterval curInterval = entry.getInterval();
            final List<BreakpointEvidence> evidenceList = entry.getValue();
            if( isValidated(entry.getValue()) || partitionCrossingChecker.onBoundary(curInterval) ) {
                // already validated (no need to mark validated again) or on partition boundary (punt for now)
                result = true;
            } else if( anyPassesFilter(evidenceList) ) {
                evidenceList.forEach(ev -> ev.setValidated(true));
                result = true;
            }

            if ( result ) {
                listItr = entry.getValue().iterator();
            }
        }
        return result;
    }

    @Override
    public BreakpointEvidence next() {
        if ( !hasNext() ) {
            throw new NoSuchElementException("No next element.");
        }
        return listItr.next();
    }

    private boolean isValidated( final List<BreakpointEvidence> evList ) {
        for ( final BreakpointEvidence ev : evList ) {
            if ( ev.isValidated() ) return true;
        }
        return false;
    }

    private boolean anyPassesFilter(final List<BreakpointEvidence> evidenceList) {
        for(final BreakpointEvidence evidence : evidenceList) {
            final double evidenceGoodProbability = predictor.predictSingle(getFeatures(evidence));
            if(evidenceGoodProbability > thresholdProbability) {
                return true;
            }
        }
        return false;
    }

    /**
     * Compute features vector for a piece of BreakpointEvidence
     */
    @VisibleForTesting
    EvidenceFeatures getFeatures(final BreakpointEvidence evidence) {
        // create new struct for these two, use CigarOperator to update if it's ReadEvidence
        final CigarQualityInfo cigarQualityInfo = new CigarQualityInfo(evidence);
        final double evidenceType = evidenceTypeMap.get(evidence.getClass().getSimpleName());
        final double mappingQuality = (double)getMappingQuality(evidence);
        final double templateSize = getTemplateSize(evidence);

        // calculate these similar to BreakpointDensityFilter, but always calculate full totals, never end early.
        final CoverageScaledOverlapInfo individualOverlapInfo = getIndividualOverlapInfo(evidence);
        final CoverageScaledOverlapInfo clusterOverlapInfo = getClusterOverlapInfo(evidence);

        // calculate properties related to overlap of intervals on the reference genome
        final double referenceGapOverlap = getGenomeIntervalsOverlap(evidence, genomeGaps, readMetadata);
        final double umapS100 = getGenomeIntervalsOverlap(evidence, umapS100Mappability, readMetadata);

        return new EvidenceFeatures(
            new double[]{
                cigarQualityInfo.basesMatched, cigarQualityInfo.referenceLength, evidenceType, mappingQuality, templateSize,
                individualOverlapInfo.numOverlap, individualOverlapInfo.overlapMappingQuality,
                individualOverlapInfo.meanOverlapMappingQuality, individualOverlapInfo.numCoherent,
                individualOverlapInfo.coherentMappingQuality,
                clusterOverlapInfo.numOverlap, clusterOverlapInfo.overlapMappingQuality,
                clusterOverlapInfo.meanOverlapMappingQuality, clusterOverlapInfo.numCoherent,
                clusterOverlapInfo.coherentMappingQuality,
                referenceGapOverlap, umapS100
            }
        );
    }

    private int getMappingQuality(final BreakpointEvidence evidence) {
        // Note: return "max" mapping quality for non-ReadEvidence. Reasoning: some features depend on sum or average of
        // read qualities. Non-ReadEvidence isn't *bad* per se, so give it a good score.
        return evidence.getMappingQuality() == null ? 60: evidence.getMappingQuality();
    }

    private double getTemplateSize(final BreakpointEvidence evidence) {
        final Integer templateSize = evidence.getTemplateSize();
        if(templateSize == null) {
            // For TemplateSizeAnomaly, return readCount scaled by meanGenomeCoverage
            final Integer readCounts = evidence.getReadCount();
            Utils.validateArg(readCounts != null,
                    "templateSize feature is only defined for ReadEvidence and TemplateSizeAnomaly");
            return (double)(readCounts) / readMetadata.getCoverage();

        } else {
            // For ReadEvidence, return templateSize as percentile of library's cumulative density function:
            final String readGroup = ((BreakpointEvidence.ReadEvidence) evidence).getReadGroup();
            final String library = readMetadata.getReadGroupToLibraryMap().get(readGroup);
            final LibraryStatistics libraryStatistics = readMetadata.getLibraryStatistics(library);
            final IntHistogram.CDF templateSizeCDF = libraryStatistics.getCDF();
            final int cdfBin = Integer.min(Math.abs(templateSize), templateSizeCDF.size() - 1);
            return templateSizeCDF.getFraction(cdfBin);
        }
    }

    private CoverageScaledOverlapInfo getIndividualOverlapInfo(final BreakpointEvidence evidence) {
        // Since overlap info will be needed for the same evidence in different contexts, it's fastest to calculate it
        // once, cache it, and then just retrieve the info each time it's needed.
        if(!rawFeatureCache.containsKey(evidence)) {
            cacheOverlapInfo(evidence);
        }
        final UnscaledOverlapInfo evidenceFeatureCache = rawFeatureCache.get(evidence);
        // Calculate the coverage scaled overlap info
        final double meanOverlapMappingQuality = ((double)evidenceFeatureCache.overlapMappingQuality)
                                               / evidenceFeatureCache.numOverlap;
        return new CoverageScaledOverlapInfo(evidenceFeatureCache.numOverlap, evidenceFeatureCache.numCoherent,
                                             evidenceFeatureCache.overlapMappingQuality, evidenceFeatureCache.coherentMappingQuality,
                                             meanOverlapMappingQuality, readMetadata.getCoverage());
    }

    private CoverageScaledOverlapInfo getClusterOverlapInfo(final BreakpointEvidence evidence) {
        int clusterNumOverlap = 0;
        int clusterNumCoherent = 0;
        int clusterOverlapMappingQuality = 0;
        int clusterCoherentMappingQuality = 0;
        double clusterMeanOverlapMappingQuality = 0.0;
        for (final Iterator<BreakpointEvidence> overlapperItr = evidenceOverlapChecker.overlappers(evidence); overlapperItr.hasNext(); ) {
            final BreakpointEvidence overlapper = overlapperItr.next();
            if (overlapper.equals(evidence)) {
                continue; // don't count self-overlap in cluster features
            }
            if(!rawFeatureCache.containsKey(overlapper)) {
                cacheOverlapInfo(overlapper);
            }
            final UnscaledOverlapInfo overlapperFeatureCache = rawFeatureCache.get(overlapper);
            clusterNumOverlap = Math.max(clusterNumOverlap, overlapperFeatureCache.numOverlap);
            clusterNumCoherent = Math.max(clusterNumCoherent, overlapperFeatureCache.numCoherent);
            clusterOverlapMappingQuality = Math.max(clusterOverlapMappingQuality, overlapperFeatureCache.overlapMappingQuality);
            clusterCoherentMappingQuality = Math.max(clusterCoherentMappingQuality, overlapperFeatureCache.coherentMappingQuality);

            final double meanOverlapMappingQuality = ((double) overlapperFeatureCache.overlapMappingQuality)
                                                    / overlapperFeatureCache.numOverlap;
            clusterMeanOverlapMappingQuality = Math.max(clusterMeanOverlapMappingQuality, meanOverlapMappingQuality);
        }

        return new CoverageScaledOverlapInfo(clusterNumOverlap, clusterNumCoherent, clusterOverlapMappingQuality,
                clusterCoherentMappingQuality, clusterMeanOverlapMappingQuality, readMetadata.getCoverage());
    }

    /**
     * For given BreakpointEvidence, calculate number of overlappers, coherent pieces of BreakpointEvidence, and
     * the sums of their mapping qualities. Cache this information so that it can be looked up when computing max or
     * average of these values over the set of overlapping evidence (as in getClusterOverlapInfo).
     */
    private void cacheOverlapInfo(final BreakpointEvidence evidence) {
        int numOverlap = 0;
        int overlapMappingQuality = 0;
        int numCoherent = 0;
        int coherentMappingQuality = 0;
        for(final EvidenceOverlapChecker.OverlapAndCoherenceIterator overlapperItr = evidenceOverlapChecker.overlappersWithCoherence(evidence);
            overlapperItr.hasNext();) {
            final ImmutablePair<BreakpointEvidence, Boolean> itrResults = overlapperItr.next();
            final BreakpointEvidence overlapper = itrResults.left;
            if(overlapper.equals(evidence)) {
                continue; // don't count self-overlap
            }
            ++numOverlap;
            final int mappingQuality = getMappingQuality(overlapper);
            overlapMappingQuality += mappingQuality;

            final boolean isCoherent = itrResults.right;
            if(isCoherent) {
                ++numCoherent;
                coherentMappingQuality += mappingQuality;
            }
        }
        rawFeatureCache.put(evidence,
                new UnscaledOverlapInfo(numOverlap, numCoherent, overlapMappingQuality, coherentMappingQuality));
    }

    private static <T> void addToTree( final SVIntervalTree<List<T>> tree,
                                       final SVInterval interval,
                                       final T value ) {
        final SVIntervalTree.Entry<List<T>> entry = tree.find(interval);
        if ( entry != null ) {
            entry.getValue().add(value);
        } else {
            final List<T> valueList = new ArrayList<>(1);
            valueList.add(value);
            tree.put(interval, valueList);
        }
    }

    private static class EvidenceOverlapChecker {
        private final SVIntervalTree<List<BreakpointEvidence>> evidenceTree;
        private final ReadMetadata readMetadata;
        private final int minEvidenceMapQ;

        EvidenceOverlapChecker(final Iterator<BreakpointEvidence> evidenceItr, final ReadMetadata readMetadata,
                               final int minEvidenceMapQ) {
            this.readMetadata = readMetadata;
            this.minEvidenceMapQ = minEvidenceMapQ;
            evidenceTree = new SVIntervalTree<>();
            Utils.stream(evidenceItr).forEach(
                    evidence -> addToTree(evidenceTree, evidence.getLocation(), evidence)
            );
        }

        private OverlapperIterator overlappers(final BreakpointEvidence evidence) {
            return new OverlapperIterator(evidence, evidenceTree);
        }

        // returns iterator to overlapping evidence, paired with a Boolean that is true if the evidence is also coherent
        private OverlapAndCoherenceIterator overlappersWithCoherence(final BreakpointEvidence evidence) {
            return new OverlapAndCoherenceIterator(evidence, evidenceTree, readMetadata, minEvidenceMapQ);
        }

        private Iterator<SVIntervalTree.Entry<List<BreakpointEvidence>>> getTreeItr() {
            return evidenceTree.iterator();
        }

        private static class OverlapperIterator  implements Iterator<BreakpointEvidence> {
            private final Iterator<SVIntervalTree.Entry<List<BreakpointEvidence>>> treeItr;
            private Iterator<BreakpointEvidence> listItr;

            OverlapperIterator(final BreakpointEvidence evidence,
                               final SVIntervalTree<List<BreakpointEvidence>> evidenceTree) {
                treeItr = evidenceTree.overlappers(evidence.getLocation());
                listItr = null;
            }

            @Override
            public boolean hasNext() {
                if ( listItr != null && listItr.hasNext() ) {
                    return true;
                }
                while(treeItr.hasNext()) {
                    listItr = treeItr.next().getValue().iterator();
                    if(listItr.hasNext()) {
                        return true;
                    }
                }
                return false;
            }

            @Override
            public BreakpointEvidence next() {
                if ( !hasNext() ) {
                    throw new NoSuchElementException("No next element.");
                }
                return listItr.next();
            }
        }

        /**
         * Implements iterator of BreakpointEvidence that overlaps a specified piece of evidence. Each call to next()
         * returns overlapping BreakpointEvidence paired with a Boolean that is true if the overlapper is also coherent
         * with the evidence.
         */
        private static class OverlapAndCoherenceIterator implements Iterator<ImmutablePair<BreakpointEvidence, Boolean>> {
            private final ReadMetadata readMetadata;
            private final int minEvidenceMapQ;
            private final Iterator<SVIntervalTree.Entry<List<BreakpointEvidence>>> treeItr;
            private Iterator<BreakpointEvidence> listItr;
            private final boolean checkCoherence;
            private final Boolean isForwardStrand;
            private final List<StrandedInterval> distalTargets;

            OverlapAndCoherenceIterator(final BreakpointEvidence evidence,
                                        final SVIntervalTree<List<BreakpointEvidence>> evidenceTree,
                                        final ReadMetadata readMetadata,
                                        final int minEvidenceMapQ) {
                this.readMetadata = readMetadata;
                this.minEvidenceMapQ = minEvidenceMapQ;
                treeItr = evidenceTree.overlappers(evidence.getLocation());
                isForwardStrand = evidence.isEvidenceUpstreamOfBreakpoint();
                checkCoherence = (isForwardStrand != null) && evidence.hasDistalTargets(readMetadata, minEvidenceMapQ);
                distalTargets = checkCoherence ?
                        new ArrayList<>(evidence.getDistalTargets(readMetadata, minEvidenceMapQ)) : null;
            }

            @Override
            public boolean hasNext() {
                if ( listItr != null && listItr.hasNext() ) {
                    return true;
                }
                while(treeItr.hasNext()) {
                    listItr = treeItr.next().getValue().iterator();
                    if(listItr.hasNext()) {
                        return true;
                    }
                }
                return false;
            }

            @Override
            public ImmutablePair<BreakpointEvidence, Boolean> next() {
                if ( !hasNext() ) {
                    throw new NoSuchElementException("No next element.");
                }
                final BreakpointEvidence overlapper = listItr.next();
                Boolean isCoherent = false;
                if(checkCoherence) {
                    Boolean overlapperStrand = overlapper.isEvidenceUpstreamOfBreakpoint();
                    if(overlapperStrand == isForwardStrand && overlapper.hasDistalTargets(readMetadata, minEvidenceMapQ)) {
                        for(final StrandedInterval distalTarget : distalTargets) {
                            for(final StrandedInterval overlapperDistalTarget : overlapper.getDistalTargets(readMetadata, minEvidenceMapQ)) {
                                if(distalTarget.getStrand() == overlapperDistalTarget.getStrand()
                                        && distalTarget.getInterval().overlaps(overlapperDistalTarget.getInterval())) {
                                    isCoherent = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                return new ImmutablePair<>(overlapper, isCoherent);
            }
        }
    }

    private static class UnscaledOverlapInfo {
        final int numOverlap;
        final int numCoherent;
        final int overlapMappingQuality;
        final int coherentMappingQuality;
        UnscaledOverlapInfo(final int numOverlap, final int numCoherent, final int overlapMappingQuality,
                            final int coherentMappingQuality) {
            this.numOverlap = numOverlap;
            this.numCoherent = numCoherent;
            this.overlapMappingQuality = overlapMappingQuality;
            this.coherentMappingQuality = coherentMappingQuality;
        }
    }

    /**
     * Class that takes raw overlap info and automatically scales it to meanGenomeCoverage, storing the result for later retrieval
     */
    private static class CoverageScaledOverlapInfo {
        final double numOverlap;
        final double overlapMappingQuality;
        final double meanOverlapMappingQuality;
        final double numCoherent;
        final double coherentMappingQuality;

        CoverageScaledOverlapInfo(final int numOverlap, final int numCoherent, final int overlapMappingQuality,
                                  final int coherentMappingQuality, final double meanOverlapMappingQuality,
                                  final double coverage) {
            this.numOverlap = ((double)numOverlap) / coverage;
            this.overlapMappingQuality = ((double)overlapMappingQuality) / coverage;
            this.numCoherent = ((double)numCoherent) / coverage;
            this.coherentMappingQuality = ((double)coherentMappingQuality) / coverage;
            this.meanOverlapMappingQuality = meanOverlapMappingQuality;
        }
    }

    private static class CigarQualityInfo {
        final int basesMatched;
        final int referenceLength;

        CigarQualityInfo(final BreakpointEvidence evidence) {
            int numMatched = 0;
            int refLength = 0;
            final String cigarString = evidence.getCigarString();
            if(cigarString != null) {
                for (final CigarElement element : TextCigarCodec.decode(cigarString).getCigarElements()) {
                    final CigarOperator op = element.getOperator();
                    if (op.consumesReferenceBases()) {
                        refLength += element.getLength();
                        if (op.consumesReadBases()) {
                            numMatched += element.getLength();
                        }
                    }
                }
            }
            basesMatched = numMatched;
            referenceLength = refLength;
        }
    }

    private static double getGenomeIntervalsOverlap(final BreakpointEvidence evidence,
                                                    final FeatureDataSource<BEDFeature> genomeIntervals,
                                                    final ReadMetadata readMetadata) {
        final SimpleInterval simpleInterval = evidence.getLocation().toSimpleInterval(readMetadata);
        double overlap = 0.0;
        for(final Iterator<BEDFeature> overlapperItr = genomeIntervals.query(simpleInterval);
            overlapperItr.hasNext();) {
            final BEDFeature overlapper = overlapperItr.next();
            // " + 1" because genome tract data is semi-closed, but BEDFeature is fully closed
            final double overlapLength = Double.min(simpleInterval.getEnd(), overlapper.getEnd()) + 1
                    - Double.max(simpleInterval.getStart(), overlapper.getStart());
            overlap += overlapLength / simpleInterval.size();
        }
        return overlap;
    }
}
