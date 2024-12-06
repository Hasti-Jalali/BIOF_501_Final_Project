#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Define processes
process preprocessing {
    container 'mednist_inference:latest'

    input:
    val inputDir

    output:
    val("/app/preprocessed_test_images.pt"), emit: preprocessedData

    script:
    """
    python /app/preprocessing.py
    """
}

process inference {
    container 'mednist_inference:latest'

    input:
    val preprocessedData

    output:
    val("/app/results.pt"), emit: inferenceResults
    

    script:
    """
    python /app/inference.py
    """
}

process finalResult {
    container 'mednist_inference:latest'

    input:
    val inferenceResults

    output:
    val("/app/test_predictions.csv"), emit :finalResultsOutput

    script:
    """
    python /app/test_results.py
    """
}

// Workflow definition
workflow {

    def curPath = file('.')

    preprocessedData = preprocessing()
    inferenceResults = trainValidation(preprocessedData)
    finalResultsOutput = testResults(inferenceResults)
}