#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Define processes

process dataGathering {
    container 'mednist_training:latest'

    input:
    val x
    val curPath

    output:
    val("${params.inputDir}"), emit: dataGatheringResult

    script:
    """
    echo $curPath
    bash /app/data_gathering.sh
    """
}

process dataPreprocessing {
    container 'mednist_training:latest'

    input:
    val inputDir

    output:
    val("${params.preprocessedDir}"), emit: preprocessedData

    script:
    """
    python /app/data_preprocessing.py
    """
}

process trainValidation {
    container 'mednist_training:latest'

    input:
    val preprocessedDir

    output:
    val("${params.trainResultsDir}"), emit: trainResults
    val("${params.modelPath}"), emit: trainedModel
    

    script:
    """
    python /app/train_validation.py
    """
}

process testResults {
    container 'mednist_training:latest'

    input:
    val preprocessedDir
    val modelPath

    output:
    val("${params.testResultsDir}"), emit :testResultsOutput

    script:
    """
    python /app/test_results.py
    """
}

// Workflow definition

workflow {
    // def inputDir = file(params.inputDir)
    // def preprocessedDir = file(params.preprocessedDir)
    def curPath = file('.')

    dataGatheringResult = dataGathering("Hi", curPath)
    preprocessedData = dataPreprocessing(dataGatheringResult)
    (trainResults, trainedModel) = trainValidation(preprocessedData)
    testResultsOutput = testResults(preprocessedData, trainedModel)

    
}