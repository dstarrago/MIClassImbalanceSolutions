/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package samplingExperimenter;

import dataPreprocessing.MWMOTE4MIL;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.filters.Filter;

/**
 * Class for testing the classifiers.
 *
 * @author Danel
 */
public class Launcher {

  private DataCollection dataCollection = new DataCollection();
  private ClassifierCollection classifierCollection = new ClassifierCollection();
  private final int folds = 5;
  private String outputDir = "C:/Users/Danel/Documents/Investigación/Proyectos/Clasificadores ensambles MIL con MISMOTE/Experimentos/";

  private void resetWeights(Instances data) {
    for (int i = 0; i < data.numInstances(); i++) {
      data.instance(i).setWeight(1);
    }
  }

  public void testDataPreprocessing() {
    String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #3/";
    String name = "Dummy1.arff";
    String fileName = path + name;
    try {
      Instances data = DataSource.read(fileName);
      resetWeights(data);
      data.setClassIndex(data.numAttributes() - 1);

      MWMOTE4MIL mwmote4mil = new MWMOTE4MIL();
      mwmote4mil.setInputFormat(data);
      mwmote4mil.setNumSynthetics(2);
      //mwmote4mil.setNearestNeighbors(1);

      Instances newData = Filter.useFilter(data, mwmote4mil);
      System.out.println(newData.toString());
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  public void testClassifier_5CV() {
    ResultsCompiler results = new ResultsCompiler("Experiment_CISS.6",
            outputDir, dataCollection, classifierCollection);
    System.out.println(results.name());
    System.out.println();
    for (int k = 0; k < classifierCollection.numClassifiers(); k++) {
      ClassifierEntry classifierEntry = classifierCollection.classifierEntry(k);
      for (int i = 0; i < dataCollection.numDatasets(); i++) {
        Classifier classifier = classifierEntry.instantiate();
        System.out.println("Classifier " + classifierEntry.name());
        DataEntry dataEntry = dataCollection.dataEntry(i);
        System.out.println("Data " + dataEntry.name());
        System.out.println();
        long trainTimeInNanoSecs = 0;
        long testTimeInNanoSecs = 0;
        try {
          Evaluation eval = null;
          for (int fold = 1; fold <= folds; fold++) {
            String trainFileName = dataEntry.trainFold(fold);
            String testFileName =  dataEntry.testFold(fold);
            Instances trainData = DataSource.read(trainFileName);
            Instances testData = DataSource.read(testFileName);
            resetWeights(trainData);
            resetWeights(testData);
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);

            long startTime = System.nanoTime();
            classifier.buildClassifier(trainData);
            long estimatedTime = System.nanoTime() - startTime;
            trainTimeInNanoSecs += estimatedTime;

            System.out.println(String.format("Built on fold %d!", fold));
            if (eval == null)
              eval = new Evaluation(trainData);
            else
              eval.setPriors(trainData);

            startTime = System.nanoTime();
            eval.evaluateModel(classifier, testData);
            estimatedTime = System.nanoTime() - startTime;
            testTimeInNanoSecs += estimatedTime;

            System.out.println(String.format("Evaluation on fold %d done!", fold));
          }

          double trainTime = trainTimeInNanoSecs / 1.0E9;       //  Train time in seconds
          double testTime = testTimeInNanoSecs / 1.0E9;         //  Test time in seconds

          results.addResult(classifierEntry, dataEntry, eval, trainTime, testTime);

          System.out.println();
          System.out.println("AUC " + eval.areaUnderROC(dataEntry.posClassLabel()));
          System.out.println("Train time " + trainTime);
          System.out.println("Test time " + testTime);
          System.out.println();
        } catch (Exception e) {
             System.err.println(e.getMessage());
        }
      }
    }
  }

  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    Launcher L = new Launcher();
    //L.testDataPreprocessing();
    L.testClassifier_5CV();
  }
}
