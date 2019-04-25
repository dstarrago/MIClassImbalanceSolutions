/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package samplingExperimenter;

import dataPreprocessing.BagSmoteKDE;
import dataPreprocessing.BagSMOTE;
import dataPreprocessing.MISMOTE;
import dataPreprocessing.MWMOTE4MIL;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import java.util.Date;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import weka.classifiers.mi.supportVector.MIRBFKernel;
//import weka.classifiers.mi.MIWrapper;
//import weka.classifiers.mi.SimpleMI;
//import weka.classifiers.mi.CitationKNN;

/**
 * Class for testing the classifiers.
 *
 * @author Danel
 */
public class LauncherPlus {

  private DataCollection dataCollection = new DataCollection();
  private ClassifierCollection classifierCollection = new ClassifierCollection();
  private final int folds = 5;
  private final int runs = 2;
//  private String outputDir = "C:/Users/Danel/Documents/Investigación/Proyectos/Clasificadores ensambles MIL con MISMOTE/Experimentos/";
  private String outputDir = "C:/Users/Danel/Documents/Investigación/Proyectos/MIL review/BOOK";
  //private String outputDir = "../Results/";

  private void resetWeights(Instances data) {
    for (int i = 0; i < data.numInstances(); i++) {
      data.instance(i).setWeight(1);
    }
  }

  public void testDataPreprocessing() {
    String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #3/";
    String name = "Dummy2.arff";
    String fileName = path + name;
    try {
      Instances data = DataSource.read(fileName);
      resetWeights(data);
      data.setClassIndex(data.numAttributes() - 1);

      MISMOTE filter = new MISMOTE();
      filter.setInputFormat(data);
      Instances newData = Filter.useFilter(data, filter);
      System.out.println(newData.toString());
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  public void testFilter_5CV() {
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss");
    String experimentName = "Experiment-" + df.format(date);
    ResultsToExcel results = new ResultsToExcel(experimentName, outputDir);
    System.out.println(results.name());
    System.out.println();
    System.out.println("*************   BagSMOTE    *************");
    System.out.println();
    try {
      results.openBook();
      Classifier[] classifier = new Classifier[classifierCollection.numClassifiers()];
      for (int k = 0; k < classifierCollection.numClassifiers(); k++) {
        ClassifierEntry classifierEntry = classifierCollection.classifierEntry(k);
        classifier[k] = classifierEntry.instantiate();
        results.addNewSheet(classifierEntry.name(), "First BagSMOTE then " + classifierEntry.name());
        //results.addNewSheet(classifierEntry.name(), classifierEntry.name() + "on imblanced data without any corrective solution");
      }
      for (int i = 0; i < dataCollection.numDatasets(); i++) {
        DataEntry dataEntry = dataCollection.dataEntry(i);
        System.out.println("Processing dataset: " + dataEntry.name());
        Evaluation[] eval = new Evaluation[classifierCollection.numClassifiers()];
        long[] trainTimeInNanoSecs = new long[classifierCollection.numClassifiers()];
        long[] testTimeInNanoSecs = new long[classifierCollection.numClassifiers()];
        for (int fold = 1; fold <= folds; fold++) {
          String trainFileName = dataEntry.trainFold(fold);
          String testFileName =  dataEntry.testFold(fold);
          Instances trainData = DataSource.read(trainFileName);
          Instances testData = DataSource.read(testFileName);
          resetWeights(trainData);
          resetWeights(testData);
          trainData.setClassIndex(trainData.numAttributes() - 1);
          testData.setClassIndex(testData.numAttributes() - 1);
          System.out.println("Load fold: " + fold);

          /**
           * Here apply the filter, sorry, yet manually
           */

          BagSMOTE filter = new BagSMOTE();
          filter.setInputFormat(trainData);
          Instances newTrainData = Filter.useFilter(trainData, filter);
          System.out.println("Applied filter: " + filter.toString());

          /**
           * Here test the filter's work
           */

          for (int k = 0; k < classifierCollection.numClassifiers(); k++) {
            ClassifierEntry classifierEntry = classifierCollection.classifierEntry(k);
            System.out.println("Training classifier " + classifierEntry.name());

            long startTime = System.nanoTime();
            classifier[k].buildClassifier(newTrainData);
            long estimatedTime = System.nanoTime() - startTime;
            trainTimeInNanoSecs[k] += estimatedTime;
            System.out.println(String.format("Built on fold %d!", fold));

            if (eval[k] == null)
              eval[k] = new Evaluation(newTrainData);
            else
              eval[k].setPriors(newTrainData);

            startTime = System.nanoTime();
            eval[k].evaluateModel(classifier[k], testData);
            estimatedTime = System.nanoTime() - startTime;
            testTimeInNanoSecs[k] += estimatedTime;

            System.out.println(String.format("Evaluation on fold %d done!", fold));
          }
        }
        for (int k = 0; k < classifierCollection.numClassifiers(); k++) {
          results.selectSheet(k);
          double trainTime = trainTimeInNanoSecs[k] / 1.0E9;       //  Train time in seconds
          double testTime = testTimeInNanoSecs[k] / 1.0E9;         //  Test time in seconds

          try {
            results.addResult(dataEntry, eval[k], trainTime, testTime);
          } catch (Exception e) {
               System.err.println("Excel writing problem: " + e.getMessage());
          }
        }
      }
      results.closeBook();
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  public void applyFilter_5CV() {
    System.out.println();
    System.out.println("*************   Aplicando método de rebalanceo de datos    *************");
    System.out.println();
    try {
      for (int i = 0; i < dataCollection.numDatasets(); i++) {
        DataEntry dataEntry = dataCollection.dataEntry(i);
        System.out.println("Processing dataset: " + dataEntry.name());
        for (int fold = 1; fold <= folds; fold++) {
          String trainFileName = dataEntry.trainFold(fold);
          Instances trainData = DataSource.read(trainFileName);
          resetWeights(trainData);
          trainData.setClassIndex(trainData.numAttributes() - 1);
          System.out.println("Load fold: " + fold);
          BagSmoteKDE filter = new BagSmoteKDE();
          //BagSMOTE filter = new BagSMOTE();
          //MISMOTE filter = new MISMOTE();
          filter.setInputFormat(trainData);
          Instances newTrainData = Filter.useFilter(trainData, filter);
          String filterName = filter.toString();
          String shortFilterName = filterName.substring(filterName.lastIndexOf(".") + 1);
          System.out.println("Applied filter: " + shortFilterName);
          String ouputStr = "_BALANCED/" + shortFilterName;
          String balancedTrainFileName = dataEntry.foldX(ouputStr, "Bal", fold);
          //String balancedTrainFileName = dataEntry.balTrainFold(fold);
          DataSink.write(balancedTrainFileName, newTrainData);
        }
      }
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  public void testClassifier_KFoldsCV() {
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss");
    String experimentName = "Experiment-" + df.format(date);
    ResultsToExcel results = new ResultsToExcel(experimentName, outputDir);
    System.out.println(results.name());
    System.out.println();
    try {
      results.openBook();
      for (int k = 0; k < classifierCollection.numClassifiers(); k++) {
        ClassifierEntry classifierEntry = classifierCollection.classifierEntry(k);
        results.addNewSheet(classifierEntry.name());
        for (int i = 0; i < dataCollection.numDatasets(); i++) {
          Classifier classifier = classifierEntry.instantiate();
          System.out.println("Classifier " + classifierEntry.name());
          DataEntry dataEntry = dataCollection.dataEntry(i);
          System.out.println("Data " + dataEntry.name());
          System.out.println();
          long trainTimeInNanoSecs = 0;
          long testTimeInNanoSecs = 0;
          Evaluation eval = null;
          for (int fold = 1; fold <= folds; fold++) {
            String trainFileName = dataEntry.trainFold(fold);
//            String trainFileName = dataEntry.balTrainFold(fold); /// Ojo trainFold(fold) -> balTrainFold(fold)
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
//            System.out.println();
//            System.out.println(classifier.toString());
//            System.out.println();
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

          try {
            results.addResult(dataEntry, eval, trainTime, testTime);
          } catch (Exception e) {
               System.err.println("Excel writing problem: " + e.getMessage());
          }

          System.out.println();
          System.out.println("AUC " + eval.areaUnderROC(dataEntry.posClassLabel()));
          System.out.println("Train time " + trainTime);
          System.out.println("Test time " + testTime);
          System.out.println();
        }
      }
      results.closeBook();
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  public void testClassifier_XTimesKFoldsCV() {
    Date date = new Date();
    DateFormat df = new SimpleDateFormat("yy.MM.dd-HH.mm.ss");
    String experimentName = "Experiment-" + df.format(date);
    ResultsToExcel results = new ResultsToExcel(experimentName, outputDir);
    System.out.println(results.name());
    System.out.println();
    try {
      results.openBook();
      for (int k = 0; k < classifierCollection.numClassifiers(); k++) {
        ClassifierEntry classifierEntry = classifierCollection.classifierEntry(k);
        results.addNewSheet(classifierEntry.name());
        for (int i = 0; i < dataCollection.numDatasets(); i++) {
          Classifier classifier = classifierEntry.instantiate();
          System.out.println("Classifier " + classifierEntry.name());
          DataEntry dataEntry = dataCollection.dataEntry(i);
          System.out.println("Data " + dataEntry.name());
          System.out.println();
          long trainTimeInNanoSecs = 0;
          long testTimeInNanoSecs = 0;
          Evaluation eval = null;
          for (int run = 1; run <= runs; run++) {
            for (int fold = 1; fold <= folds; fold++) {
              String trainFileName = dataEntry.trainFold(run, fold);
              String testFileName =  dataEntry.testFold(run, fold);
              System.out.println(trainFileName);
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

              System.out.println(String.format("Built on run %d fold %d!", run, fold));
              if (eval == null)
                eval = new Evaluation(trainData);
              else
                eval.setPriors(trainData);

              startTime = System.nanoTime();
              eval.evaluateModel(classifier, testData);
              estimatedTime = System.nanoTime() - startTime;
              testTimeInNanoSecs += estimatedTime;

              System.out.println(String.format("Evaluation on run %d fold %d done!", run, fold));
            }
          }

          double trainTime = trainTimeInNanoSecs / 1.0E9;       //  Train time in seconds
          double testTime = testTimeInNanoSecs / 1.0E9;         //  Test time in seconds

          try {
            results.addResult(dataEntry, eval, trainTime, testTime);
          } catch (Exception e) {
               System.err.println("Excel writing problem: " + e.getMessage());
          }

          System.out.println();
          System.out.println("AUC " + eval.areaUnderROC(dataEntry.posClassLabel()));
          System.out.println("Train time " + trainTime);
          System.out.println("Test time " + testTime);
          System.out.println();
        }
      }
      results.closeBook();
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }


    private Instances bagsmote(int n, Instances D) throws Exception {
      BagSMOTE bagsmote = new BagSMOTE();
      bagsmote.setInputFormat(D);
      bagsmote.setNumSynthetics(n);
      Instances syn = Filter.useFilter(D, bagsmote);
      return syn;
    }

    private Instances mismote(int n, Instances D) throws Exception {
      MISMOTE mismote = new MISMOTE();
      mismote.setInputFormat(D);
      mismote.SetNumSynthetics(n);
      Instances syn = Filter.useFilter(D, mismote);
      return syn;
    }

    private Instances mwmote4mil(int n, Instances D) throws Exception {
      MWMOTE4MIL mwmote4mil = new MWMOTE4MIL();
      mwmote4mil.setInputFormat(D);
      mwmote4mil.setNumSynthetics(n);
      Instances syn = Filter.useFilter(D, mwmote4mil);
      return syn;
    }

  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    LauncherPlus L = new LauncherPlus();
    L.testClassifier_KFoldsCV();
    //L.testDataPreprocessing();
    //L.testFilter_5CV();
    //L.applyFilter_5CV();
  }
}
