package org.broadinstitute.hellbender.tools.spark.sv.integration;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.utils.io.IOUtils;
import org.broadinstitute.hellbender.utils.test.ArgumentsBuilder;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.broadinstitute.hellbender.utils.test.IntegrationTestSpec;
import org.broadinstitute.hellbender.utils.test.MiniClusterUtils;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Integration test on the SV pipeline as it exists right now [2017-03-06]
 */
public class FindBreakpointEvidenceSparkIntegrationTest extends CommandLineProgramTest {

    private static final File expectedSAMfile = IOUtils.getPath(SVIntegrationTestDataProvider.EXPECTED_ALIGNED_CONTIGS).toFile();

    private static final class FindBreakpointEvidenceSparkIntegrationTestArgs {
        final String bamLoc;
        final String kmerIgnoreListLoc;
        final String alignerRefIndexImgLoc;
        final String outputDir;
        final float bamCoverage;

        FindBreakpointEvidenceSparkIntegrationTestArgs(final String bamLoc,
                                                       final String kmerIgnoreListLoc,
                                                       final String alignerRefIndexImgLoc,
                                                       final String outputDir,
                                                       final float bamCoverage) {
            this.bamLoc = bamLoc;
            this.kmerIgnoreListLoc = kmerIgnoreListLoc;
            this.alignerRefIndexImgLoc = alignerRefIndexImgLoc;
            this.outputDir = outputDir;
            this.bamCoverage = bamCoverage;
        }

        String getCommandLine() {
            return  " -I " + bamLoc +
                    " -O "                    + outputDir + "/assemblies.sam" +
                    " --aligner-index-image " + alignerRefIndexImgLoc +
                    " --kmers-to-ignore " + kmerIgnoreListLoc +
                    " --breakpoint-intervals " + outputDir + "/intervals" +
                    " --fastq-dir "            + outputDir + "/fastq" +
                    " --target-link-file "      + outputDir + "/targetLinks.bedpe" +
                    " --min-evidence-count " + 15 / bamCoverage +
                    " --min-coherent-evidence-count " + 7 / bamCoverage +
                    " --sv-evidence-filter-type " + "DENSITY";
        }

        @Override
        public String toString() {
            return "FindBreakpointEvidenceSparkIntegrationTestArgs{" +
                    "bam-loc='" + bamLoc + '\'' +
                    ", kmer-ignore-list-loc='" + kmerIgnoreListLoc + '\'' +
                    ", aligner-ref-index-img-loc='" + alignerRefIndexImgLoc + '\'' +
                    ", output-dir='" + outputDir + '\'' +
                    '}';
        }
    }

    @DataProvider(name = "findBreakpointEvidenceSparkIntegrationTest")
    public Object[][] createTestData() throws IOException {

        List<Object[]> tests = new ArrayList<>();
        final File tempDirNew = BaseTest.createTempDir("forNew");
        tempDirNew.deleteOnExit();
        Files.createDirectories(Paths.get(tempDirNew.getAbsolutePath()+"/fastq"));
        tests.add(new Object[]{new FindBreakpointEvidenceSparkIntegrationTestArgs(SVIntegrationTestDataProvider.TEST_BAM,
                SVIntegrationTestDataProvider.KMER_KILL_LIST, SVIntegrationTestDataProvider.ALIGNER_INDEX_IMG,
                tempDirNew.getAbsolutePath(), SVIntegrationTestDataProvider.TEST_BAM_COVERAGE)});

        return tests.toArray(new Object[][]{});
    }

    @Test(dataProvider = "findBreakpointEvidenceSparkIntegrationTest", groups = "sv")
    public void testFindBreakpointRunnableLocal(final FindBreakpointEvidenceSparkIntegrationTestArgs params) throws IOException {

        final List<String> args = Arrays.asList( new ArgumentsBuilder().add(params.getCommandLine()).getArgsArray() );
        runCommandLine(args);

        final File actualSAMfile = new File(params.outputDir, "assemblies.sam");
        IntegrationTestSpec.assertEqualTextFiles(actualSAMfile, expectedSAMfile);
    }

    @Test(dataProvider = "findBreakpointEvidenceSparkIntegrationTest", groups = "sv")
    public void testFindBreakpointRunnableMiniCluster(final FindBreakpointEvidenceSparkIntegrationTestArgs params) throws Exception {

        MiniClusterUtils.runOnIsolatedMiniCluster(cluster -> {

            final List<String> argsToBeModified = Arrays.asList( new ArgumentsBuilder().add(params.getCommandLine()).getArgsArray() );
            final Path workingDirectory = MiniClusterUtils.getWorkingDir(cluster);

            // copy local data to minicluster file system and update args
            changeArgCopyFromLocal(argsToBeModified, "-I", new Path(workingDirectory, "hdfs.bam"), cluster);
            changeArgCopyFromLocal(argsToBeModified, "--kmers-to-ignore", new Path(workingDirectory, "dummy.kill.kmers"), cluster);

            // outputs, prefix with hdfs address
            changeArg(argsToBeModified, "-O",
                    new Path(workingDirectory, "assemblies.sam").toUri().toString());
            changeArg(argsToBeModified, "--breakpoint-intervals",
                    new Path(workingDirectory, "intervals").toUri().toString());
            changeArg(argsToBeModified, "--fastq-dir",
                    new Path(workingDirectory, "fastq").toUri().toString());

            runCommandLine(argsToBeModified);

            final File actualSAMfile = BaseTest.createTempFile("assemblies", ".sam");
            cluster.getFileSystem().copyToLocalFile(new Path(workingDirectory, "assemblies.sam"), new Path(actualSAMfile.toURI()));
            IntegrationTestSpec.assertEqualTextFiles(actualSAMfile, expectedSAMfile);
        });
    }

    @Test(dataProvider = "findBreakpointEvidenceSparkIntegrationTest", groups = "sv")
    public void testFindBreakpointRunnableXGBoostMiniCluster(final FindBreakpointEvidenceSparkIntegrationTestArgs params) throws Exception {

        MiniClusterUtils.runOnIsolatedMiniCluster(cluster -> {

            final List<String> argsToBeModified = new ArrayList<>(Arrays.asList( new ArgumentsBuilder().add(params.getCommandLine()).getArgsArray() ));
            final Path workingDirectory = MiniClusterUtils.getWorkingDir(cluster);

            // copy local data to minicluster file system and update args
            changeArgCopyFromLocal(argsToBeModified, "-I", new Path(workingDirectory, "hdfs.bam"), cluster);
            changeArgCopyFromLocal(argsToBeModified, "--kmers-to-ignore", new Path(workingDirectory, "dummy.kill.kmers"), cluster);

            // outputs, prefix with hdfs address
            changeArg(argsToBeModified, "-O",
                    new Path(workingDirectory, "assemblies.sam").toUri().toString());
            changeArg(argsToBeModified, "--breakpoint-intervals",
                    new Path(workingDirectory, "intervals").toUri().toString());
            changeArg(argsToBeModified, "--fastq-dir",
                    new Path(workingDirectory, "fastq").toUri().toString());

            // use xgboost classifier
            changeArg(argsToBeModified, "--sv-evidence-filter-type", "XGBOOST");
            addArg(argsToBeModified, "--sv-genome-gaps-file", SVIntegrationTestDataProvider.TEST_GENOME_GAPS_FILE);
            addArg(argsToBeModified, "--sv-genome-umap-s100-file", SVIntegrationTestDataProvider.TEST_GENOME_UMAP100_FILE);

            new IntegrationTestSpec(String.join(" ", argsToBeModified), SVIntegrationTestDataProvider.dummyExpectedFileNames)
                    .executeTest("testFindBreakpointEvidenceSparkRunnableXGBoostMiniCluster-", this);
        });
    }

    static private void changeArg(final List<String> argsToBeModified, final String arg, final String newVal) {
        final int idx = argsToBeModified.indexOf(arg);
        argsToBeModified.set(idx + 1, newVal);
    }

    static private void changeArgCopyFromLocal(final List<String> argsToBeModified, final String arg, final Path newPath,
                                               final MiniDFSCluster cluster) throws Exception {
        final int idx = argsToBeModified.indexOf(arg);
        final File file = new File(argsToBeModified.get(idx + 1));
        cluster.getFileSystem().copyFromLocalFile(new Path(file.toURI()), newPath);
        argsToBeModified.set(idx + 1, newPath.toString());
    }

    static private void addArg(final List<String> argsToBeModified, final String arg, final String newVal) {
        argsToBeModified.add(arg);
        argsToBeModified.add(newVal);
    }
}
