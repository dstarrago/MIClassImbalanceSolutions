/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.classifiers.Evaluation;
import weka.core.converters.ArffSaver;

/**
 *
 * @author Danel
 */
public class ResultsCompiler {

  private String name;
  private String outputDir;
  private Instances results;
  private ArffSaver saver;
  private DataCollection dataCollection;
  private ClassifierCollection classifierCollection;

  public ResultsCompiler(String name, String outputDir, 
          DataCollection dataCollection, ClassifierCollection classifierCollection) {
    this.name = name;
    this.outputDir = outputDir;
    this.dataCollection = dataCollection;
    this.classifierCollection =  classifierCollection;
    results = buildInstances();
    initializeSaver();
  }

  public ResultsCompiler(String name, String outputDir, DataCollection dataCollection) {
    this(name, outputDir, dataCollection, null);
  }

  public ResultsCompiler(String name, String outputDir, ClassifierCollection classifierCollection) {
    this(name, outputDir, null, classifierCollection);
  }

  public ResultsCompiler(String name, String outputDir) {
    this(name, outputDir, null, null);
  }

  private List<String> classifierList() {
    if (classifierCollection == null) return null;
    List<String> list = new ArrayList<String>();
    for (int i = 0; i < classifierCollection.numClassifiers(); i++) {
      list.add(classifierCollection.classifierEntry(i).name());
    }
    return list;
  }

  private List<String> dataCollection() {
    if (dataCollection == null) return null;
    List<String> list = new ArrayList<String>();
    for (int i = 0; i < dataCollection.numDatasets(); i++) {
      list.add(dataCollection.dataEntry(i).name());
    }
    return list;
  }

  private List<String> validation() {
    if (dataCollection == null) return null;
    List<String> list = new ArrayList<String>();
    for (int i = 0; i < dataCollection.numDatasets(); i++) {
      String s = String.format(DataEntry.VALIDATION_MASK, dataCollection.dataEntry(i).numFolds());
      if (!list.contains(s))
        list.add(s);
    }
    return list;
  }

  private Instances buildInstances() {
    String relationName = name;
    ArrayList<Attribute> attInfo = new ArrayList<Attribute>();

    /* 00 */ attInfo.add(new Attribute("Classifier", classifierList())); // classifier name
    /* 01 */ attInfo.add(new Attribute("Data", dataCollection()));       // data name
    /* 02 */ attInfo.add(new Attribute("Validation", validation()));     // validation scheme
    /* 03 */ attInfo.add(new Attribute("Pos_Class"));                    // positive class label

    /* 04 */ attInfo.add(new Attribute("TP"));               // true positives
    /* 05 */ attInfo.add(new Attribute("FP"));               // false positives
    /* 06 */ attInfo.add(new Attribute("FN"));               // false negatives
    /* 07 */ attInfo.add(new Attribute("TN"));               // true negatives

    /* 08 */ attInfo.add(new Attribute("Acc"));               // accuracy
    /* 09 */ attInfo.add(new Attribute("kappa"));             // kappa
    /* 10 */ attInfo.add(new Attribute("AUC"));               // auc for the positive class
    /* 11 */ attInfo.add(new Attribute("precision"));         // precision for the positive class
    /* 12 */ attInfo.add(new Attribute("recall"));            // recall for the positive class
    /* 13 */ attInfo.add(new Attribute("F1"));                // F1 for the positive class

    /* 15 */ attInfo.add(new Attribute("Train_Time"));        // Time elapsed in training
    /* 16 */ attInfo.add(new Attribute("Test_Time"));         // Time elapsed in testing

    int numResults = 1;
    if (dataCollection != null && classifierCollection == null) numResults = dataCollection.numDatasets();
    if (dataCollection == null && classifierCollection != null) numResults = classifierCollection.numClassifiers();
    if (dataCollection != null && classifierCollection != null)
      numResults = dataCollection.numDatasets() * classifierCollection.numClassifiers();

    Instances instances = new Instances(relationName, attInfo, numResults);
    return instances;
  }

  public void addResult(ClassifierEntry classifier, DataEntry data, Evaluation eval,
          double trainTime, double testTime) {

    double[] attVals = new double[results.numAttributes()];

    attVals[0] = results.attribute(0).indexOfValue(classifier.name());
    attVals[1] = results.attribute(1).indexOfValue(data.name());
    attVals[2] = results.attribute(2).indexOfValue(String.format(DataEntry.VALIDATION_MASK, data.numFolds()));
    attVals[3] = data.posClassLabel();

    attVals[4] = eval.numTruePositives(data.posClassLabel());
    attVals[5] = eval.numFalsePositives(data.posClassLabel());
    attVals[6] = eval.numFalseNegatives(data.posClassLabel());
    attVals[7] = eval.numTrueNegatives(data.posClassLabel());

    attVals[8] = eval.pctCorrect();
    attVals[9] = eval.kappa();
    attVals[10] = eval.areaUnderROC(data.posClassLabel());
    attVals[11] = eval.precision(data.posClassLabel());
    attVals[12] = eval.recall(data.posClassLabel());
    attVals[13] = eval.fMeasure(data.posClassLabel());

    attVals[14] = trainTime;
    attVals[15] = testTime;

    Instance inst = new DenseInstance(1, attVals);
    results.add(inst);
    saveResult(inst);
  }
  
  private void initializeSaver() {
    /**
     * Set the file address
     */
    File file = new File(outputDir, name + ".arff");
    /**
     * Try to create saver object
     */
    try {
      saver = new ArffSaver();
      saver.setFile(file);
      saver.setInstances(results);
      saver.setRetrieval(ArffSaver.INCREMENTAL);
      } catch (Exception e) {
           System.err.println(e.getMessage());
     }

  }
  
  private void saveResult(Instance inst) {
    try {
      saver.writeIncremental(inst);
      saver.getWriter().flush();
      } catch (Exception e) {
           System.err.println(e.getMessage());
     }
  }

  public String name() {
    return name;
  }
  
  public String outputDir() {
    return outputDir;
  }

}
