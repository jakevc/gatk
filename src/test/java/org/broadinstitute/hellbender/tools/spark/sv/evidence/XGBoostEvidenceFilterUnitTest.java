package org.broadinstitute.hellbender.tools.spark.sv.evidence;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import htsjdk.samtools.SAMFileHeader;
import htsjdk.samtools.SAMReadGroupRecord;
import htsjdk.samtools.SAMSequenceDictionary;
import htsjdk.samtools.SAMSequenceRecord;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.broadinstitute.hellbender.GATKBaseTest;
import org.broadinstitute.hellbender.engine.spark.SparkContextFactory;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.spark.sv.StructuralVariationDiscoveryArgumentCollection;
import org.broadinstitute.hellbender.tools.spark.utils.IntHistogram;
import org.testng.AssertJUnit;
import org.testng.annotations.Test;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class XGBoostEvidenceFilterUnitTest extends GATKBaseTest {
    private static final String testAccuracyDataJsonFile = publicTestDir + "sv_classifier_test_data.json";
    private static final String classifierModelFile = "/large/sv_evidence_classifier.bin";
    private static final String resourceClassifierModelFile = "gatk-resources::" + classifierModelFile;
    private static final String localClassifierModelFile
            = new File(publicMainResourcesDir, classifierModelFile).getAbsolutePath();
    private static final String testFeaturesJsonFile = publicTestDir + "sv_features_test_data.json";
    private static final double probabilityTol = 2.0e-4;
    private static final double featuresTol = 1.0e-6;

    private final ClassifierAccuracyData classifierAccuracyData = new ClassifierAccuracyData(testAccuracyDataJsonFile);
    private final double[] predictYProbaSerial = predictProba(
            XGBoostEvidenceFilter.loadPredictor(localClassifierModelFile), classifierAccuracyData.features
    );
    private static final FeaturesTestData featuresTestData = new FeaturesTestData(testFeaturesJsonFile);

    private static final StructuralVariationDiscoveryArgumentCollection.FindBreakpointEvidenceSparkArgumentCollection params
            = new StructuralVariationDiscoveryArgumentCollection.FindBreakpointEvidenceSparkArgumentCollection();

    private static final SAMFileHeader artificialSamHeader = initSAMFileHeader();
    private static final String readGroupName = "Pond-Testing";
    private static final String DEFAULT_SAMPLE_NAME = "SampleX";
    public static final ReadMetadata readMetadata = initMetadata();
    private static final PartitionCrossingChecker emptyCrossingChecker = new PartitionCrossingChecker();
    private final List<BreakpointEvidence> evidenceList = Arrays.stream(featuresTestData.stringReps)
            .map(strRep -> BreakpointEvidence.fromStringRep(strRep, readMetadata)).collect(Collectors.toList());

    private static SAMFileHeader initSAMFileHeader() {
        final SAMFileHeader samHeader = createArtificialSamHeader();
        SAMReadGroupRecord readGroup = new SAMReadGroupRecord(readGroupName);
        readGroup.setSample(DEFAULT_SAMPLE_NAME);
        samHeader.addReadGroup(readGroup);
        return samHeader;
    }

    /**
     * Create synthetic SAM Header comptible with genome tracts (e.g. has all the primary contigs)
     * Use genome gaps file to enforce this.
     */
    public static SAMFileHeader createArtificialSamHeader() {
        Comparator<String> contigComparator = new Comparator<String>() {
            @Override public int compare(String s1, String s2) {
                return getKey(s1).compareTo(getKey(s2));
            }

            String getKey(final String contig) {
                try {
                    final int contigNum = Integer.parseInt(contig.substring(3));
                    return "chr" + (contigNum < 10 ? "0" + contigNum : contigNum);
                } catch(NumberFormatException e) {
                    if(contig.equals("chrX") || contig.equals("chrY")) {
                        return contig;
                    } else {
                        return "zzz" + contig;
                    }
                }
            }
        };

        SortedMap<String,Integer> contigLengthMap = new TreeMap<>(contigComparator);
        final String svGenomeGapsFile = params.svGenomeGapsFile;
        int lineNumber = -1;
        try(final InputStream inputStream = XGBoostEvidenceFilter.getInputStream(svGenomeGapsFile)) {
            final BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while((line = reader.readLine()) != null) {
                ++lineNumber;
                if(line.startsWith("#")) {
                    continue;
                }
                final String[] splitLine = line.split("\t");
                final String contig = splitLine[0];
                final Integer end = Integer.parseInt(splitLine[2]) + 1;
                if(!contigLengthMap.containsKey(contig) || contigLengthMap.get(contig) < end) {
                    contigLengthMap.put(contig, end);
                }
            }
        } catch(IOException e) {
            throw new GATKException("Unable to read genome gaps file \"" + svGenomeGapsFile + "\" line " + lineNumber
                                    + ": " + e.getMessage());
        }

        SAMFileHeader header = new SAMFileHeader();
        header.setSortOrder(SAMFileHeader.SortOrder.coordinate);
        SAMSequenceDictionary dict = new SAMSequenceDictionary();
        // make up some sequence records
        for (SortedMap.Entry<String,Integer> contigEntry : contigLengthMap.entrySet()) {
            final String contig = contigEntry.getKey();
            final int chromosomeSize = contigEntry.getValue();
            SAMSequenceRecord rec = new SAMSequenceRecord(contig, chromosomeSize);
            rec.setSequenceLength(chromosomeSize);
            dict.addSequence(rec);
        }
        header.setSequenceDictionary(dict);
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
        final float[] cdfFractions = new float[cumulativeCounts.length];
        final long totalObservations = cumulativeCounts[cumulativeCounts.length - 1];
        for(int index = 0; index < cdfFractions.length; ++index) {
            cdfFractions[index] = cumulativeCounts[index] / (float)totalObservations;
        }
        return new IntHistogram.CDF(cdfFractions, totalObservations);
    }

    @Test(groups = "sv")
    protected void testLocalXGBoostClassifierAccuracy() {
        // check accuracy: predictions are same as classifierAccuracyData up to tolerance
        AssertJUnit.assertArrayEquals("Probabilities predicted by classifier do not match saved correct answers",
                predictYProbaSerial, classifierAccuracyData.yProba, probabilityTol);
    }

    @Test(groups = "sv")
    protected void testLocalXGBoostClassifierSpark() {
        final Predictor localPredictor = XGBoostEvidenceFilter.loadPredictor(localClassifierModelFile);
        // get spark ctx
        final JavaSparkContext ctx = SparkContextFactory.getTestSparkContext();
        // parallelize classifierAccuracyData to RDD
        JavaRDD<FVec> testFeaturesRdd = ctx.parallelize(Arrays.asList(classifierAccuracyData.features));
        // predict in parallel
        JavaDoubleRDD predictYProbaRdd
                = testFeaturesRdd.mapToDouble(f -> localPredictor.predictSingle(f, false, 0));
        // pull back to local array
        final double[] predictYProbaSpark = predictYProbaRdd.collect()
                .stream().mapToDouble(Double::doubleValue).toArray();
        // check probabilities from spark are identical to serial
        AssertJUnit.assertArrayEquals("Probabilities predicted in spark context differ from serial",
                predictYProbaSpark, predictYProbaSerial, 0.0);
    }

    @Test(groups = "sv")
    protected void testResourceXGBoostClassifier() {
        // load classifier from resource
        final Predictor resourcePredictor = XGBoostEvidenceFilter.loadPredictor(resourceClassifierModelFile);
        final double[] resourceYProba = predictProba(resourcePredictor, classifierAccuracyData.features);
        // check that predictions from resource are identical to local
        AssertJUnit.assertArrayEquals("Predictions via loading predictor from resource is not identical to local file",
                resourceYProba, predictYProbaSerial, 0.0);
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

            final BreakpointEvidence convertedEvidence = BreakpointEvidence.fromStringRep(stringRep, readMetadata);
            final String convertedRep = convertedEvidence.stringRep(readMetadata, params.minEvidenceMapQ);
            AssertJUnit.assertEquals("BreakpointEvidence.fromStringRep does not invert BreakpointEvidence.stringRep",
                    stringRep.trim(), convertedRep.trim());
            final EvidenceFeatures calcFVec = evidenceFilter.getFeatures(evidence);
            AssertJUnit.assertArrayEquals("Features calculated by XGBoostEvidenceFilter don't match expected features",
                    fVec.getValues(), calcFVec.getValues(), featuresTol);
        }
    }

    @Test(groups = "sv")
    protected void testFilter() {
        final List<BreakpointEvidence> expectedPassed = new ArrayList<>();
        int index = 0;
        for(final BreakpointEvidence evidence : evidenceList) {
            final double proba = featuresTestData.proba[index];
            if(proba > params.svEvidenceFilterThresholdProbability) {
                expectedPassed.add(evidence);
            }
            index += 1;
        }

        final XGBoostEvidenceFilter evidenceFilter = new XGBoostEvidenceFilter(
                evidenceList.iterator(), readMetadata, params, emptyCrossingChecker
        );
        final List<BreakpointEvidence> passedEvidence = new ArrayList<>();
        evidenceFilter.forEachRemaining(passedEvidence::add);

        AssertJUnit.assertEquals("Evidence passed by XGBoostEvidenceFilter not the same as expected",
                expectedPassed, passedEvidence);
    }

    private static double[] predictProba(final Predictor predictor, final FVec[] testFeatures) {
        final int numData = testFeatures.length;
        final double[] yProba = new double[numData];
        for(int rowIndex = 0; rowIndex < numData; ++rowIndex) {
            yProba[rowIndex] = predictor.predictSingle(testFeatures[rowIndex], false,0);
        }
        return yProba;
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
                    fVec.set_value(columnIndex, valueNode.asDouble());
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

    private static class ClassifierAccuracyData extends JsonMatrixLoader {
        final EvidenceFeatures[] features;
        final double[] yProba;

        ClassifierAccuracyData(final String jsonFileName) {
            try(final InputStream inputStream = new FileInputStream(jsonFileName)) {
                final JsonNode testDataNode = new ObjectMapper().readTree(inputStream);
                features = getFVecArrayFromJsonNode(testDataNode.get("features"));
                yProba = getDoubleArrayFromJsonNode(testDataNode.get("proba"));
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
        final double[] proba;
        final float coverage;
        final long[] template_size_cumulative_counts;

        FeaturesTestData(final String jsonFileName) {
            try(final InputStream inputStream = new FileInputStream(jsonFileName)) {
                final JsonNode testDataNode = new ObjectMapper().readTree(inputStream);
                features = getFVecArrayFromJsonNode(testDataNode.get("features"));
                stringReps = getStringArrayFromJsonNode(testDataNode.get("string_reps"));
                proba = getDoubleArrayFromJsonNode(testDataNode.get("proba"));
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
