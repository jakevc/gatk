package org.broadinstitute.hellbender.tools.spark.sv.evidence;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import htsjdk.samtools.SAMFileHeader;
import htsjdk.samtools.SAMReadGroupRecord;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.broadinstitute.hellbender.GATKBaseTest;
import org.broadinstitute.hellbender.engine.spark.SparkContextFactory;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.spark.sv.discovery.inference.CpxSVInferenceTestUtils;
import org.broadinstitute.hellbender.tools.spark.sv.utils.SVInterval;
import org.broadinstitute.hellbender.tools.spark.sv.utils.StrandedInterval;
import org.broadinstitute.hellbender.tools.spark.utils.IntHistogram;
import org.broadinstitute.hellbender.utils.Utils;
import org.testng.Assert;
import org.testng.annotations.Test;
import org.broadinstitute.hellbender.tools.spark.sv.StructuralVariationDiscoveryArgumentCollection.FindBreakpointEvidenceSparkArgumentCollection;


import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import static org.broadinstitute.hellbender.utils.Utils.validateArg;

public class XGBoostEvidenceFilterUnitTest extends GATKBaseTest {
    private static final String SV_EVIDENCE_TEST_DIR = toolsTestDir + "spark/sv/evidence/FindBreakpointEvidenceSpark/";
    private static final String testAccuracyDataJsonFile = SV_EVIDENCE_TEST_DIR + "sv_classifier_test_data.json";
    private static final String classifierModelFile = "/large/sv_evidence_classifier.bin";
    private static final String localClassifierModelFile
            = new File(publicMainResourcesDir, classifierModelFile).getAbsolutePath();
    private static final String testFeaturesJsonFile = SV_EVIDENCE_TEST_DIR + "sv_features_test_data.json";
    private static final double probabilityTol = 2.0e-3;
    private static final double featuresTol = 1.0e-5;
    private static final String SV_GENOME_UMAP_S100_FILE = SV_EVIDENCE_TEST_DIR + "hg38_umap_s100.bed.gz";
    private static final String SV_GENOME_GAPS_FILE = SV_EVIDENCE_TEST_DIR + "hg38_gaps.bed.gz";

    private static final ClassifierAccuracyData classifierAccuracyData = new ClassifierAccuracyData(testAccuracyDataJsonFile);
    private static final double[] predictedProbabilitySerial = predictProbability(
            XGBoostEvidenceFilter.loadPredictor(localClassifierModelFile), classifierAccuracyData.features
    );
    private static final FeaturesTestData featuresTestData = new FeaturesTestData(testFeaturesJsonFile);

    private static final FindBreakpointEvidenceSparkArgumentCollection params = initParams();

    private static final SAMFileHeader artificialSamHeader = initSAMFileHeader();
    private static final String readGroupName = "Pond-Testing";
    private static final String DEFAULT_SAMPLE_NAME = "SampleX";
    private static final ReadMetadata readMetadata = initMetadata();
    private static final PartitionCrossingChecker emptyCrossingChecker = new PartitionCrossingChecker();
    private static final BreakpointEvidenceFactory breakpointEvidenceFactory = new BreakpointEvidenceFactory(readMetadata);
    private static final List<BreakpointEvidence> evidenceList = Arrays.stream(featuresTestData.stringReps)
            .map(breakpointEvidenceFactory::fromStringRep).collect(Collectors.toList());

    private static FindBreakpointEvidenceSparkArgumentCollection initParams() {
        final FindBreakpointEvidenceSparkArgumentCollection params = new FindBreakpointEvidenceSparkArgumentCollection();
        params.svGenomeUmapS100File = SV_GENOME_UMAP_S100_FILE;
        params.svGenomeGapsFile = SV_GENOME_GAPS_FILE;
        return params;
    }

    private static SAMFileHeader initSAMFileHeader() {
        final SAMFileHeader samHeader = createArtificialSamHeader();
        SAMReadGroupRecord readGroup = new SAMReadGroupRecord(readGroupName);
        readGroup.setSample(DEFAULT_SAMPLE_NAME);
        samHeader.addReadGroup(readGroup);
        return samHeader;
    }

    /**
     * Create synthetic SAM Header comptible with genome tracts (e.g. has all the primary contigs)
     */
    public static SAMFileHeader createArtificialSamHeader() {
        final SAMFileHeader header = new SAMFileHeader();
        header.setSortOrder(SAMFileHeader.SortOrder.coordinate);
        header.setSequenceDictionary(CpxSVInferenceTestUtils.bareBoneHg38SAMSeqDict);
        return header;
    }

    private static ReadMetadata initMetadata() {
        final ReadMetadata.PartitionBounds[] partitionBounds = new ReadMetadata.PartitionBounds[3];
        partitionBounds[0] = new ReadMetadata.PartitionBounds(0, 1, 0, 10000, 9999);
        partitionBounds[1] = new ReadMetadata.PartitionBounds(0, 10001, 0, 20000, 9999);
        partitionBounds[2] = new ReadMetadata.PartitionBounds(0, 20001, 0, 30000, 9999);
        return new ReadMetadata(Collections.emptySet(), artificialSamHeader,
                new LibraryStatistics(cumulativeCountsToCDF(featuresTestData.template_size_cumulative_counts),
                        60000000000L, 600000000L, 1200000000000L, 3000000000L),
                partitionBounds, 100, 10, featuresTestData.coverage);
    }

    private static IntHistogram.CDF cumulativeCountsToCDF(final long[] cumulativeCounts) {
        final long totalObservations = cumulativeCounts[cumulativeCounts.length - 1];
        final float[] cdfFractions = new float[cumulativeCounts.length];
        for(int index = 0; index < cdfFractions.length; ++index) {
            cdfFractions[index] = cumulativeCounts[index] / (float)totalObservations;
        }
        return new IntHistogram.CDF(cdfFractions, totalObservations);
    }

    @Test(groups = "sv")
    protected void testLocalXGBoostClassifierAccuracy() {
        // check accuracy: predictions are same as classifierAccuracyData up to tolerance
        assertArrayEquals(predictedProbabilitySerial, classifierAccuracyData.probability, probabilityTol, "Probabilities predicted by classifier do not match saved correct answers"
        );
    }

    @Test(groups = "sv")
    protected void testLocalXGBoostClassifierSpark() {
        final Predictor localPredictor = XGBoostEvidenceFilter.loadPredictor(localClassifierModelFile);
        // get spark ctx
        final JavaSparkContext ctx = SparkContextFactory.getTestSparkContext();
        // parallelize classifierAccuracyData to RDD
        JavaRDD<FVec> testFeaturesRdd = ctx.parallelize(Arrays.asList(classifierAccuracyData.features));
        // predict in parallel
        JavaDoubleRDD predictedProbabilityRdd
                = testFeaturesRdd.mapToDouble(f -> localPredictor.predictSingle(f, false, 0));
        // pull back to local array
        final double[] predictedProbabilitySpark = predictedProbabilityRdd.collect()
                .stream().mapToDouble(Double::doubleValue).toArray();
        // check probabilities from spark are identical to serial
        assertArrayEquals(predictedProbabilitySpark, predictedProbabilitySerial, 0.0, "Probabilities predicted in spark context differ from serial"
        );
    }

    @Test(groups = "sv")
    protected void testResourceXGBoostClassifier() {
        // load classifier from resource
        final Predictor resourcePredictor = XGBoostEvidenceFilter.loadPredictor(null);
        final double[] predictedProbabilityResource = predictProbability(resourcePredictor, classifierAccuracyData.features);
        // check that predictions from resource are identical to local
        assertArrayEquals(predictedProbabilityResource, predictedProbabilitySerial, 0.0, "Predictions via loading predictor from resource is not identical to local file"
        );
    }

    @Test(groups = "sv")
    protected void testFeatureConstruction() {
        final XGBoostEvidenceFilter evidenceFilter = new XGBoostEvidenceFilter(
                evidenceList.iterator(), readMetadata, params, emptyCrossingChecker
        );
        for(int ind = 0; ind < featuresTestData.stringReps.length; ++ind) {
            final BreakpointEvidence evidence = evidenceList.get(ind);
            final String stringRep = featuresTestData.stringReps[ind];
            final EvidenceFeatures fVec = featuresTestData.features[ind];

            final BreakpointEvidence convertedEvidence = breakpointEvidenceFactory.fromStringRep(stringRep);
            final String convertedRep = convertedEvidence.stringRep(readMetadata, params.minEvidenceMapQ);
            Assert.assertEquals(stringRep.trim(), convertedRep.trim(),
                    "BreakpointEvidenceFactory.fromStringRep does not invert BreakpointEvidence.stringRep");
            final EvidenceFeatures calcFVec = evidenceFilter.getFeatures(evidence);
            assertArrayEquals(fVec.getValues(), calcFVec.getValues(), featuresTol, "Features calculated by XGBoostEvidenceFilter don't match expected features"
            );
        }
    }

    @Test(groups = "sv")
    protected void testFilter() {
        final List<BreakpointEvidence> expectedPassed = new ArrayList<>();
        int index = 0;
        for(final BreakpointEvidence evidence : evidenceList) {
            final double probability = featuresTestData.probability[index];
            if(probability > params.svEvidenceFilterThresholdProbability) {
                expectedPassed.add(evidence);
            }
            index += 1;
        }

        final XGBoostEvidenceFilter evidenceFilter = new XGBoostEvidenceFilter(
                evidenceList.iterator(), readMetadata, params, emptyCrossingChecker
        );
        final List<BreakpointEvidence> passedEvidence = new ArrayList<>();
        evidenceFilter.forEachRemaining(passedEvidence::add);

        Assert.assertEquals(expectedPassed, passedEvidence,
                "Evidence passed by XGBoostEvidenceFilter not the same as expected");
    }

    private static void assertArrayEquals(final double[] expecteds, final double[] actuals, final double tol,
                                          final String message) {
        Assert.assertEquals(expecteds.length, actuals.length, "Lengths not equal: " + message);
        for(int index = 0; index < expecteds.length; ++index) {
            Assert.assertEquals(expecteds[index], actuals[index], tol, "at index=" + index + ": " + message);
        }
    }

    private static double[] predictProbability(final Predictor predictor, final FVec[] testFeatures) {

        return Arrays.stream(testFeatures).mapToDouble(
                features -> predictor.predictSingle(features, false, 0)
        ).toArray();
    }

    static class JsonMatrixLoader {
        static EvidenceFeatures[] getFVecArrayFromJsonNode(final JsonNode matrixNode) {
            if(!matrixNode.has("__class__")) {
                throw new IllegalArgumentException("JSON node does not store python matrix data");
            }
            String matrixClass = matrixNode.get("__class__").asText();
            switch(matrixClass) {
                case "pandas.DataFrame":
                    return getFVecArrayFromPandasJsonNode(matrixNode.get("data"));
                case "numpy.array":
                    return getFVecArrayFromNumpyJsonNode(matrixNode.get("data"));
                default:
                    throw new IllegalArgumentException("JSON node has __class__ = " + matrixClass
                            + "which is not a supported matrix type");
            }
        }

        private static EvidenceFeatures[] getFVecArrayFromNumpyJsonNode(final JsonNode dataNode) {
            if(!dataNode.isArray()) {
                throw new IllegalArgumentException("dataNode does not encode a valid numpy array");
            }
            final int numRows = dataNode.size();
            final EvidenceFeatures[] matrix = new EvidenceFeatures[numRows];
            if (numRows == 0) {
                return matrix;
            }
            matrix[0] = new EvidenceFeatures(getDoubleArrayFromJsonArrayNode(dataNode.get(0)));
            final int numColumns = matrix[0].length();
            for (int row = 1; row < numRows; ++row) {
                matrix[row] = new EvidenceFeatures(getDoubleArrayFromJsonArrayNode(dataNode.get(row)));
                final int numRowColumns = matrix[row].length();
                if (numRowColumns != numColumns) {
                    throw new IllegalArgumentException("Rows in JSONArray have different lengths.");
                }
            }
            return matrix;
        }

        private static EvidenceFeatures[] getFVecArrayFromPandasJsonNode(final JsonNode dataNode) {
            if(!dataNode.isObject()) {
                throw new IllegalArgumentException("dataNode does not encode a valid pandas DataFrame");
            }
            final int numColumns = dataNode.size();
            if(numColumns == 0) {
                return new EvidenceFeatures[0];
            }

            final String firstColumnName = dataNode.fieldNames().next();
            final int numRows = getColumnArrayNode(dataNode.get(firstColumnName)).size();
            final EvidenceFeatures[] matrix = new EvidenceFeatures[numRows];
            if (numRows == 0) {
                return matrix;
            }
            // allocate each EvidenceFeatures in matrix
            for(int rowIndex = 0; rowIndex < numRows; ++rowIndex) {
                matrix[rowIndex] = new EvidenceFeatures(numColumns);
            }
            int columnIndex = 0;
            for(final Iterator<Map.Entry<String, JsonNode>> fieldIter = dataNode.fields(); fieldIter.hasNext();) {
                // loop over columns
                final Map.Entry<String, JsonNode> columnEntry = fieldIter.next();
                final JsonNode columnArrayNode = getColumnArrayNode(columnEntry.getValue());
                if(columnArrayNode.size() != numRows) {
                    throw new IllegalArgumentException("field " + columnEntry.getKey() + " has "
                            + columnArrayNode.size() + " rows (expected " + numRows + ")");
                }
                // for each FVec in matrix, assign feature from this column
                int rowIndex = 0;
                for(final JsonNode valueNode: columnArrayNode) {
                    final EvidenceFeatures fVec = matrix[rowIndex];
                    fVec.setValue(columnIndex, valueNode.asDouble());
                    ++rowIndex;
                }
                ++columnIndex;
            }
            return matrix;
        }

        private static JsonNode getColumnArrayNode(final JsonNode columnNode) {
            return columnNode.has("values") ? columnNode.get("values") : columnNode.get("codes");
        }

        static double[] getDoubleArrayFromJsonNode(final JsonNode vectorNode) {
            if(!vectorNode.has("__class__")) {
                return getDoubleArrayFromJsonArrayNode(vectorNode);
            }
            final String vectorClass = vectorNode.get("__class__").asText();
            switch(vectorClass) {
                case "pandas.Series":
                    return getDoubleArrayFromJsonArrayNode(getColumnArrayNode(vectorNode));
                case "numpy.array":
                    return getDoubleArrayFromJsonArrayNode(vectorNode.get("data"));
                default:
                    throw new IllegalArgumentException("JSON node has __class__ = " + vectorClass
                            + "which is not a supported matrix type");
            }
        }

        private static double [] getDoubleArrayFromJsonArrayNode(final JsonNode arrayNode) {
            if(!arrayNode.isArray()) {
                throw new IllegalArgumentException("JsonNode does not contain an Array");
            }
            final int numData = arrayNode.size();
            final double[] data = new double[numData];
            int ind = 0;
            for(final JsonNode valueNode : arrayNode) {
                data[ind] = valueNode.asDouble();
                ++ind;
            }
            return data;
        }

        static long[] getLongArrayFromJsonNode(final JsonNode vectorNode) {
            if(!vectorNode.has("__class__")) {
                return getLongArrayFromJsonArrayNode(vectorNode);
            }
            final String vectorClass = vectorNode.get("__class__").asText();
            switch(vectorClass) {
                case "pandas.Series":
                    return getLongArrayFromJsonArrayNode(getColumnArrayNode(vectorNode));
                case "numpy.array":
                    return getLongArrayFromJsonArrayNode(vectorNode.get("data"));
                default:
                    throw new IllegalArgumentException("JSON node has __class__ = " + vectorClass
                            + "which is not a supported matrix type");
            }
        }

        private static long [] getLongArrayFromJsonArrayNode(final JsonNode arrayNode) {
            if(!arrayNode.isArray()) {
                throw new IllegalArgumentException("JsonNode does not contain an Array");
            }
            final int numData = arrayNode.size();
            final long[] data = new long[numData];
            int ind = 0;
            for(final JsonNode valueNode : arrayNode) {
                data[ind] = valueNode.asInt();
                ++ind;
            }
            return data;
        }

        static String[] getStringArrayFromJsonNode(final JsonNode arrayNode) {
            if(!arrayNode.isArray()) {
                throw new IllegalArgumentException("JsonNode does not contain an Array");
            }
            final int numStrings = arrayNode.size();
            final String[] stringArray = new String[numStrings];
            int ind = 0;
            for(final JsonNode stringNode : arrayNode) {
                stringArray[ind] = stringNode.asText();
                ++ind;
            }
            return stringArray;
        }
    }

    private static class BreakpointEvidenceFactory {
        final ReadMetadata readMetadata;

        BreakpointEvidenceFactory(final ReadMetadata readMetadata) {
            this.readMetadata = readMetadata;
        }

        /**
         * Returns BreakpointEvidence constructed from string representation. Used to reconstruct BreakpointEvidence for
         * unit tests. It is intended for stringRep() to be an inverse of this function, but not the other way around. i.e.
         *      fromStringRep(strRep, readMetadata).stringRep(readMetadata, minEvidenceMapQ) == strRep
         * but it may be the case that
         *      fromStringRep(evidence.stringRep(readMetadata, minEvidenceMapQ), readMetadata) != evidence
         */
        BreakpointEvidence fromStringRep(final String strRep) {
            final String[] words = strRep.split("\t");

            final SVInterval location = locationFromStringRep(words[0]);

            final int weight = Integer.parseInt(words[1]);

            final String evidenceType = words[2];
            if(evidenceType.equals("TemplateSizeAnomaly")) {
                final int readCount = Integer.parseInt(words[4]);
                return new BreakpointEvidence.TemplateSizeAnomaly(location, weight, readCount);
            } else {
                final List<StrandedInterval> distalTargets = words[3].isEmpty() ? new ArrayList<>()
                        : Arrays.stream(words[3].split(";")).map(BreakpointEvidenceFactory::strandedLocationFromStringRep)
                        .collect(Collectors.toList());
                validateArg(distalTargets.size() <= 1, "BreakpointEvidence must have 0 or 1 distal targets");
                final String[] templateParts = words[4].split("/");
                final String templateName = templateParts[0];
                final TemplateFragmentOrdinal fragmentOrdinal;
                if(templateParts.length <= 1) {
                    fragmentOrdinal = TemplateFragmentOrdinal.UNPAIRED;
                } else switch (templateParts[1]) {
                    case "0":
                        fragmentOrdinal = TemplateFragmentOrdinal.PAIRED_INTERIOR;
                        break;
                    case "1":
                        fragmentOrdinal = TemplateFragmentOrdinal.PAIRED_FIRST;
                        break;
                    case "2":
                        fragmentOrdinal = TemplateFragmentOrdinal.PAIRED_SECOND;
                        break;
                    case "?":
                        fragmentOrdinal = TemplateFragmentOrdinal.PAIRED_UNKNOWN;
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown Template Fragment Ordinal: /" + templateParts[1]);
                }
                final boolean forwardStrand = words[5].equals("1");
                final int templateSize = Integer.parseInt(words[6]);
                final String cigarString = words[7];
                final int mappingQuality = Integer.parseInt(words[8]);
                final String readGroup = "Pond-Testing"; // for now, just fake this, only for testing.
                final boolean validated = false;


                final SVInterval target;
                final boolean targetForwardStrand;
                final int targetQuality;
                switch(distalTargets.size()) {
                    case 0:
                        target = new SVInterval(0, 0, 0);
                        targetForwardStrand = false;
                        targetQuality = -1;
                        break;
                    case 1:
                        target = distalTargets.get(0).getInterval();
                        targetForwardStrand = distalTargets.get(0).getStrand();
                        targetQuality = Integer.MAX_VALUE;
                        break;
                    default:
                        throw new IllegalArgumentException("BreakpointEvidence must have <= 1 distal target");
                }

                switch(evidenceType) {
                    case "SplitRead":
                        // NOTE: can't identically reconstruct original values, but can make self-consistent values that reproduce
                        // the known distal targets. Make plausible cigar strings, primaryAlignmentClippedAtStart and
                        // primaryAlignmentForwardStrand that are compatible with dumped distal targets.
                        final String tagSA = distalTargets.isEmpty() ? null : distalTargets.stream().map(this::distalTargetToTagSA).collect(Collectors.joining());
                        return new BreakpointEvidence.SplitRead(location, weight, templateName, fragmentOrdinal, validated,
                                forwardStrand, cigarString, mappingQuality, templateSize, readGroup,
                                forwardStrand, forwardStrand, tagSA);

                    case "LargeIndel":
                        Utils.validateArg(distalTargets.isEmpty(), "LargeIndel should have no distal targets");
                        return new BreakpointEvidence.LargeIndel(location, weight, templateName, fragmentOrdinal, validated,
                                forwardStrand, cigarString, mappingQuality, templateSize, readGroup);

                    case "MateUnmapped":
                        Utils.validateArg(distalTargets.isEmpty(), "MateUnmapped should have no distal targets");
                        return new BreakpointEvidence.MateUnmapped(location, weight, templateName, fragmentOrdinal, validated,
                                forwardStrand, cigarString, mappingQuality, templateSize, readGroup);

                    case "InterContigPair":
                        return new BreakpointEvidence.InterContigPair(
                                location, weight, templateName, fragmentOrdinal, validated, forwardStrand, cigarString,
                                mappingQuality, templateSize, readGroup, target, targetForwardStrand, targetQuality
                        );

                    case "OutiesPair":
                        return new BreakpointEvidence.OutiesPair(
                                location, weight, templateName, fragmentOrdinal, validated, forwardStrand, cigarString,
                                mappingQuality, templateSize, readGroup, target, targetForwardStrand, targetQuality
                        );

                    case "SameStrandPair":
                        return new BreakpointEvidence.SameStrandPair(
                                location, weight, templateName, fragmentOrdinal, validated, forwardStrand, cigarString,
                                mappingQuality, templateSize, readGroup, target, targetForwardStrand, targetQuality
                        );

                    case "WeirdTemplateSize":
                        return new BreakpointEvidence.WeirdTemplateSize(
                                location, weight, templateName, fragmentOrdinal, validated, forwardStrand, cigarString,
                                mappingQuality, templateSize, readGroup, target, targetForwardStrand, targetQuality
                        );
                    default:
                        throw new IllegalArgumentException("Unknown BreakpointEvidence type: " + evidenceType);
                }
            }
        }

        private String distalTargetToTagSA(final StrandedInterval distalTarget) {
            final String contigName = readMetadata.getContigName(distalTarget.getInterval().getContig());
            final boolean isForwardStrand = distalTarget.getStrand();
            final int referenceLength = distalTarget.getInterval().getLength();
            final int pos = distalTarget.getInterval().getEnd() - 1 - BreakpointEvidence.SplitRead.UNCERTAINTY;
            final int start = isForwardStrand ? pos - referenceLength: pos;
            final int clipLength = readMetadata.getAvgReadLen() - referenceLength;
            final String cigar = referenceLength >= readMetadata.getAvgReadLen() ? referenceLength + "M"
                    : (isForwardStrand ? referenceLength + "M" + clipLength + "S"
                    : clipLength + "S" + referenceLength + "M");
            final int mapq = Integer.MAX_VALUE;
            final int mismatches = 0;
            final String[] tagParts = new String[] {contigName, String.valueOf(start), isForwardStrand ? "+": "-",
                    cigar, String.valueOf(mapq), String.valueOf(mismatches)};
            return String.join(",", tagParts) + ";";
        }

        private static SVInterval locationFromStringRep(final String locationStr) {
            final String[] locationParts = locationStr.split("[\\[\\]:]");
            validateArg(locationParts.length >= 2, "Could not parse SVInterval from string");
            final int contig = Integer.parseInt(locationParts[0]);
            final int start = Integer.parseInt(locationParts[1]);
            final int end = Integer.parseInt(locationParts[2]);
            return new SVInterval(contig, start, end);
        }

        private static StrandedInterval strandedLocationFromStringRep(final String locationStr) {
            final String[] locationParts = locationStr.split("[\\[\\]:]");
            validateArg(locationParts.length == 4, "Could not parse StrandedInterval from string");
            final int contig = Integer.parseInt(locationParts[0]);
            final int start = Integer.parseInt(locationParts[1]);
            final int end = Integer.parseInt(locationParts[2]);
            final boolean strand = locationParts[3].equals("1");
            return new StrandedInterval(new SVInterval(contig, start, end), strand);
        }

    }

    private static class ClassifierAccuracyData extends JsonMatrixLoader {
        final EvidenceFeatures[] features;
        final double[] probability;

        ClassifierAccuracyData(final String jsonFileName) {
            try(final InputStream inputStream = new FileInputStream(jsonFileName)) {
                final JsonNode testDataNode = new ObjectMapper().readTree(inputStream);
                features = getFVecArrayFromJsonNode(testDataNode.get("features"));
                probability = getDoubleArrayFromJsonNode(testDataNode.get("proba"));
            } catch(Exception e) {
                throw new GATKException(
                        "Unable to load classifier test data from " + jsonFileName + ": " + e.getMessage()
                );
            }
        }
    }

    private static class FeaturesTestData extends JsonMatrixLoader {
        final EvidenceFeatures[] features;
        final String[] stringReps;
        final double[] probability;
        final float coverage;
        final long[] template_size_cumulative_counts;

        FeaturesTestData(final String jsonFileName) {
            try(final InputStream inputStream = new FileInputStream(jsonFileName)) {
                final JsonNode testDataNode = new ObjectMapper().readTree(inputStream);
                features = getFVecArrayFromJsonNode(testDataNode.get("features"));
                stringReps = getStringArrayFromJsonNode(testDataNode.get("string_reps"));
                probability = getDoubleArrayFromJsonNode(testDataNode.get("proba"));
                coverage = (float)testDataNode.get("coverage").asDouble();
                template_size_cumulative_counts = getLongArrayFromJsonNode(
                        testDataNode.get("template_size_cumulative_counts")
                );
            } catch(Exception e) {
                throw new GATKException(
                        "Unable to load classifier test data from " + jsonFileName + ": " + e.getMessage()
                );
            }
        }
    }


}
