# This workflow takes an input CRAM to call variants with HaplotypeCaller and filter the calls with a neural net
# The CRAM could be generated by the single-sample pipeline
# (https://github.com/gatk-workflows/broad-prod-wgs-germline-snps-indels/blob/master/PairedEndSingleSampleWf.wdl)
workflow Cram2FilteredVcf {
    File input_cram                  # Aligned CRAM file
    File reference_fasta 
    File reference_dict
    File reference_fasta_index
    File resource_fofn               # File of VCF file names of resources of known SNPs and INDELs, (e.g. mills, gnomAD)
    File resource_fofn_index         # File of VCF file indices of resources
    File? architecture_json          # Neural Net configuration for CNNScoreVariants
    File? architecture_hd5           # Pre-Trained weights and architecture for CNNScoreVariants
    Int? inference_batch_size        # Batch size for python in CNNScoreVariants
    Int? transfer_batch_size         # Batch size for java in CNNScoreVariants
    Int? intra_op_threads            # Tensorflow threading within nodes
    Int? inter_op_threads            # Tensorflow threading between nodes
    String output_prefix             # Identifying string for this run will be used to name all output files
    String? tensor_type              # What kind of tensors the Neural Net expects (e.g. reference, read_tensor)
    String info_key                  # The score key for the info field of the vcf (e.g. CNN_1D, CNN_2D)
    String tranches                  # Filtering threshold(s) in terms of sensitivity to overlapping known variants in resources
    File? gatk_override
    String gatk_docker
    String cnn_gatk_docker
    File calling_intervals
    Int scatter_count                # Number of shards for parallelization of HaplotypeCaller and CNNScoreVariants
    String extra_args                # Extra arguments for HaplotypeCaller

    # Runtime parameters
    Int? mem_gb
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 

    call CramToBam {
        input:
          reference_fasta = reference_fasta,
          reference_dict = reference_dict,
          reference_fasta_index = reference_fasta_index,
          cram_file = input_cram,
          output_prefix = output_prefix,
          disk_space_gb = disk_space_gb,
          preemptible_attempts = preemptible_attempts
    }

    call SplitIntervals {
        input:
            gatk_override = gatk_override,
            scatter_count = scatter_count,
            intervals = calling_intervals,
            ref_fasta = reference_fasta,
            ref_dict = reference_dict,
            ref_fai = reference_fasta_index,
            gatk_docker = gatk_docker
    }

    scatter (calling_interval in SplitIntervals.interval_files) {

        call RunHC4 {
            input:
                input_bam = CramToBam.output_bam,
                input_bam_index = CramToBam.output_bam_index,
                reference_fasta = reference_fasta,
                reference_dict = reference_dict,
                reference_fasta_index = reference_fasta_index,
                output_prefix = output_prefix,
                interval_list = calling_interval,
                gatk_docker = gatk_docker,
                gatk_override = gatk_override,
                preemptible_attempts = preemptible_attempts,
                extra_args = extra_args,
                disk_space_gb = disk_space_gb
        }

        call CNNScoreVariants {
            input:
                input_vcf = RunHC4.raw_vcf,
                input_vcf_index = RunHC4.raw_vcf_index,
                bam_file = RunHC4.bamout,
                bam_file_index = RunHC4.bamout_index,
                architecture_json = architecture_json,
                architecture_hd5 = architecture_hd5,
                reference_fasta = reference_fasta,
                tensor_type = tensor_type,
                inference_batch_size = inference_batch_size,
                transfer_batch_size = transfer_batch_size,
                intra_op_threads = intra_op_threads,
                inter_op_threads = inter_op_threads,
                reference_dict = reference_dict,
                reference_fasta_index = reference_fasta_index,               
                output_prefix = output_prefix,
                interval_list = calling_interval,
                gatk_override = gatk_override,
                cnn_gatk_docker = cnn_gatk_docker,
                preemptible_attempts = preemptible_attempts,
                mem_gb = mem_gb
        }
    }

    call MergeVCFs as MergeVCF_HC4 {
        input: 
            input_vcfs = CNNScoreVariants.cnn_annotated_vcf,
            output_prefix = output_prefix,
            gatk_override = gatk_override,
            preemptible_attempts = preemptible_attempts,
            gatk_docker = gatk_docker
    }

    call FilterVariantTranches {
        input:
            input_vcf = MergeVCF_HC4.merged_vcf,
            input_vcf_index = MergeVCF_HC4.merged_vcf_index,
            resource_fofn = resource_fofn,
            resource_fofn_index = resource_fofn_index,
            output_prefix = output_prefix,
            tranches = tranches,
            info_key = info_key,
            gatk_override = gatk_override,
            preemptible_attempts = preemptible_attempts,
            gatk_docker = gatk_docker
    }

    call SamtoolsMergeBAMs {
        input:
            input_bams = RunHC4.bamout,
            output_prefix = output_prefix
    }

    output {
        MergeVCF_HC4.*
        SamtoolsMergeBAMs.*
        FilterVariantTranches.*
    }
}

task CramToBam {
    File reference_fasta
    File reference_fasta_index
    File reference_dict
    File cram_file
    String output_prefix

    # Runtime parameters
    Int? mem_gb
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 

    # You may have to change the following two parameter values depending on the task requirements
    Int default_ram_mb = 16000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).
    Int default_disk_space_gb = 200

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

command <<<
  ls -ltr ${cram_file} ${reference_fasta} &&
  echo "ls (1): complete" &&
  samtools view -h -T ${reference_fasta} ${cram_file} |
  samtools view -b -o ${output_prefix}.bam - &&
  echo "samtools view: complete" &&
  ls -ltr . &&
  echo "ls (2): complete" &&
  samtools index -b ${output_prefix}.bam &&
  echo "samtools index: complete" &&
  ls -ltr . &&
  echo "ls (3): complete" &&
  mv ${output_prefix}.bam.bai ${output_prefix}.bai &&
  echo "mv: complete" &&
  ls -ltr . &&
  echo "ls (4): complete"
  >>>
  runtime {
    docker: "broadinstitute/genomes-in-the-cloud:2.1.1"
    memory: machine_mem + " MB"
    # Note that the space before SSD and HDD should be included.
    disks: "local-disk " + select_first([disk_space_gb, default_disk_space_gb]) + " SSD"
    preemptible: select_first([preemptible_attempts, 3])
    cpu: select_first([cpu, 1])  
  }

  output {
    File output_bam = "${output_prefix}.bam"
    File output_bam_index = "${output_prefix}.bai"
  }
}

task RunHC4 {
    File input_bam
    File input_bam_index
    File reference_fasta
    File reference_dict
    File reference_fasta_index
    String output_prefix
    File interval_list
    String extra_args
    File? gatk_override

    # Runtime parameters
    Int? mem_gb
    String gatk_docker
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 

    # You may have to change the following two parameter values depending on the task requirements
    Int default_ram_mb = 8000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).
    Int default_disk_space_gb = 200

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

    command {
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk_override}

        #gatk --java-options "-Xmx${command_mem}m" \ 
        java "-Xmx${command_mem}m" -jar ${gatk_override} \
        HaplotypeCaller \
        -R ${reference_fasta} \
        -I ${input_bam} \
        -O ${output_prefix}_hc4.vcf.gz \
        -L ${interval_list} \
        -bamout ${output_prefix}_bamout.bam \
        ${extra_args}
    }

    output {
        File bamout = "${output_prefix}_bamout.bam"
        File bamout_index = "${output_prefix}_bamout.bai"
        File raw_vcf = "${output_prefix}_hc4.vcf.gz"
        File raw_vcf_index = "${output_prefix}_hc4.vcf.gz.tbi"
    }
    runtime {
        docker: gatk_docker
        memory: machine_mem + " MB"
        # Note that the space before SSD and HDD should be included.
        disks: "local-disk " + select_first([disk_space_gb, default_disk_space_gb]) + " SSD"
        preemptible: select_first([preemptible_attempts, 3])
        cpu: select_first([cpu, 1])
        zones: "us-east4-a"
    }
}

task CNNScoreVariants {
    String input_vcf
    File input_vcf_index
    File reference_fasta
    File reference_dict
    File reference_fasta_index
    String? bam_file
    String? bam_file_index
    File? architecture_json
    File? architecture_hd5
    String? tensor_type
    String output_prefix
    Int? inference_batch_size
    Int? transfer_batch_size
    Int? intra_op_threads
    Int? inter_op_threads
    File interval_list
    File? gatk_override

    # Runtime parameters
    Int? mem_gb
    String cnn_gatk_docker
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 

    String output_vcf = "${output_prefix}_cnn_annotated.vcf.gz"

    # You may have to change the following two parameter values depending on the task requirements
    Int default_ram_mb = 16000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).
    Int default_disk_space_gb = 200

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

command <<<
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk_override}

        #gatk --java-options "-Xmx${command_mem}m" \
        java "-Xmx${command_mem}m" -jar ${gatk_override} \
        CNNScoreVariants \
        ${"-I " + bam_file} \
        -R ${reference_fasta} \
        -V ${input_vcf} \
        -O ${output_vcf} \
        -L ${interval_list} \
        ${"--architecture " + architecture_json} \
        ${"--tensor-type " + tensor_type} \
        ${"--inference-batch-size " + inference_batch_size} \
        ${"--transfer-batch-size " + transfer_batch_size} \
        ${"--intra-op-threads " + intra_op_threads} \
        ${"--inter-op-threads " + inter_op_threads}
>>>

  runtime {
    docker: cnn_gatk_docker
    memory: machine_mem + " MB"
    # Note that the space before SSD and HDD should be included.
    disks: "local-disk " + select_first([disk_space_gb, default_disk_space_gb]) + " SSD"
    preemptible: select_first([preemptible_attempts, 3])
    cpu: select_first([cpu, 1])
    minCpuPlatform: "Intel Haswell"
    zones: "us-east4-a"
    bootDiskSizeGb: "16"
  }

  output {
    File cnn_annotated_vcf = "${output_vcf}"
    File cnn_annotated_vcf_index = "${output_vcf}.tbi"
  }
}


task FilterVariantTranches {
    String input_vcf
    File input_vcf_index
    File resource_fofn
    File resource_fofn_index
    Array[File] resource_files = read_lines(resource_fofn)
    Array[File] resource_files_index = read_lines(resource_fofn_index)
    String output_prefix
    String tranches
    String info_key
    File? gatk_override

    # Runtime parameters
    Int? mem_gb
    String gatk_docker
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu

    String output_vcf = "${output_prefix}_filtered.vcf.gz"

    # You may have to change the following two parameter values depending on the task requirements
    Int default_ram_mb = 16000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).
    Int default_disk_space_gb = 200

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

command <<<
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk_override}

        #gatk --java-options "-Xmx${command_mem}m" \

        java "-Xmx${command_mem}m" -jar ${gatk_override} \
        FilterVariantTranches \
        -V ${input_vcf} \
        --output ${output_vcf} \
        -resource ${sep=" -resource " resource_files} \
        -info-key ${info_key} \
        ${tranches}


>>>

  runtime {
    docker: gatk_docker
    memory: machine_mem + " MB"
    # Note that the space before SSD and HDD should be included.
    disks: "local-disk " + select_first([disk_space_gb, default_disk_space_gb]) + " SSD"
    preemptible: select_first([preemptible_attempts, 3])
    cpu: select_first([cpu, 1])
    minCpuPlatform: "Intel Haswell"
  }

  output {
    File cnn_filtered_vcf = "${output_vcf}"
    File cnn_filtered_vcf_index = "${output_vcf}.tbi"
  }
}


task SplitIntervals {
    # inputs
    File? intervals
    File ref_fasta
    File ref_fai
    File ref_dict
    Int scatter_count
    String? split_intervals_extra_args

    File? gatk_override

    # runtime
    String gatk_docker
    Int? mem
    Int? preemptible_attempts
    Int? disk_space
    Int? cpu

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 3500
    Int command_mem = machine_mem - 500

    command {
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk_override}

        mkdir interval-files
        #gatk --java-options "-Xmx${command_mem}m" SplitIntervals \
        java "-Xmx${command_mem}m" -jar ${gatk_override} SplitIntervals \
            -R ${ref_fasta} \
            ${"-L " + intervals} \
            -scatter ${scatter_count} \
            -O interval-files \
            ${split_intervals_extra_args}
        cp interval-files/*.intervals .
    }

    runtime {
        docker: gatk_docker
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
        preemptible: select_first([preemptible_attempts, 10])
        cpu: select_first([cpu, 1])
    }

    output {
        Array[File] interval_files = glob("*.intervals")
    }
}

task MergeVCFs {
    # inputs
    Array[File] input_vcfs
    String output_prefix

    File? gatk_override

    # runtime
    String gatk_docker
    Int? mem
    Int? preemptible_attempts
    Int? disk_space
    Int? cpu

    String output_vcf = "${output_prefix}_cnn_scored.vcf.gz"

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 3500
    Int command_mem = machine_mem - 1000

    # using MergeVcfs instead of GatherVcfs so we can create indices
    # WARNING 2015-10-28 15:01:48 GatherVcfs  Index creation not currently supported when gathering block compressed VCFs.
    command {
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk_override}
        #gatk --java-options "-Xmx${command_mem}m" MergeVcfs \
        java "-Xmx${command_mem}m" -jar ${gatk_override} MergeVcfs \
        -I ${sep=' -I ' input_vcfs} -O "${output_vcf}"
    }

    runtime {
        docker: gatk_docker
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
        preemptible: select_first([preemptible_attempts, 10])
        cpu: select_first([cpu, 1])
    }

    output {
        File merged_vcf = "${output_vcf}"
        File merged_vcf_index = "${output_vcf}.tbi"
    }
}

task SamtoolsMergeBAMs {
    Array[File] input_bams
    String output_prefix
    command {
        samtools merge ${output_prefix}_bamout.bam ${sep=' ' input_bams}
        samtools index ${output_prefix}_bamout.bam ${output_prefix}_bamout.bai
    }

    output {
        File bamout = "${output_prefix}_bamout.bam"
        File bamout_index = "${output_prefix}_bamout.bai"
    }

  runtime {
    docker: "broadinstitute/genomes-in-the-cloud:2.1.1"
    memory: "16 GB"
    disks: "local-disk 400 HDD"
  }    
}
