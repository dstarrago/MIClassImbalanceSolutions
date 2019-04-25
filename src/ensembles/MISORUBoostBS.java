/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    MIRUBoostM2.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package ensembles;

import dataPreprocessing.MwmoteDataManager;
import Utils.CosineDist;
import dataPreprocessing.BagSMOTE;
import dataPreprocessing.MISMOTE;
import dataPreprocessing.MWMOTE4MIL;
import dataPreprocessing.MismoteDataManager;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.Sourcable;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;

/**
 <!-- globalinfo-start -->
 * Class for boosting a nominal class classifier using the Adaboost M1 method. Only nominal class problems can be tackled. Often dramatically improves performance, but sometimes overfits.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Yoav Freund, Robert E. Schapire: Experiments with a new boosting algorithm. In: Thirteenth International Conference on Machine Learning, San Francisco, 148-156, 1996.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Freund1996,
 *    address = {San Francisco},
 *    author = {Yoav Freund and Robert E. Schapire},
 *    booktitle = {Thirteenth International Conference on Machine Learning},
 *    pages = {148-156},
 *    publisher = {Morgan Kaufmann},
 *    title = {Experiments with a new boosting algorithm},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -P &lt;num&gt;
 *  Percentage of weight mass to base training on.
 *  (default 100, reduce to around 90 speed up)</pre>
 * 
 * <pre> -Q
 *  Use resampling for boosting.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.DecisionStump)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.DecisionStump:
 * </pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Dánel S. Tarragó (danels@uclv.edu.cu)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision: 8034 $ 
 */
public class MISORUBoostBS
  extends RandomizableIteratedSingleClassifierEnhancer 
  implements WeightedInstancesHandler, Sourcable, TechnicalInformationHandler {

  /** for serialization */
  static final long serialVersionUID = -7378107808933117974L;
  
  /** Max num iterations tried to find classifier with non-zero error. */ 
  private static int MAX_NUM_RESAMPLING_ITERATIONS = 10;
  
  /** Array for storing the weights for the votes. */
  protected double [] m_Betas;

  /** The number of successfully generated base classifiers. */
  protected int m_NumIterationsPerformed;

  /** Weight Threshold. The percentage of weight mass used in training */
  protected int m_WeightThreshold = 100;

  /** Use boosting with reweighting? */
  protected boolean m_UseResampling;

  /** The number of classes */
  protected int m_NumClasses;
  
  /** a ZeroR model in case no model can be built from the data */
  protected Classifier m_ZeroR;

  /**
   * Per cent of the total desired training examples to be represented by the minority class
   */
  private double gamma = 50;
  
  /**
   * Degree of disturbance of gamma
   */
  private int sigma = 5;

  /**
   * Negative reduction percent
   */
  private int beta = 0;

  /**
   * Optimal minimum dataset size
   */
  private int eta = 150;

  /**
   * Whether bootstrap the negative class
   */
  private boolean negBootstrap = false;

  /**
   * Whether bootstrap the positive class
   */
  private boolean posBootstrap = false;

  /**
   * Calculates and stores data relative to the training dataset. Data are
   * subsequently used for classification and data processing algorithms.
   */
//  private MwmoteDataManager dm;
  private MismoteDataManager dm;

  /**
   * Constructor.
   */
  public MISORUBoostBS() {
    m_Classifier = new weka.classifiers.mi.MITI();
//    m_Classifier = new weka.classifiers.mi.CitationKNN();
//    ((weka.classifiers.mi.CitationKNN)m_Classifier).setHDRank(2);
//    ((weka.classifiers.mi.CitationKNN)m_Classifier).setNumCiters(3);
//    ((weka.classifiers.mi.CitationKNN)m_Classifier).setNumReferences(3);
  }
    
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
 
    return "Class for boosting a nominal class classifier using the Adaboost "
      + "M1 method. Only nominal class problems can be tackled. Often "
      + "dramatically improves performance, but sometimes overfits.\n\n"
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Yoav Freund and Robert E. Schapire");
    result.setValue(Field.TITLE, "Experiments with a new boosting algorithm");
    result.setValue(Field.BOOKTITLE, "Thirteenth International Conference on Machine Learning");
    result.setValue(Field.YEAR, "1996");
    result.setValue(Field.PAGES, "148-156");
    result.setValue(Field.PUBLISHER, "Morgan Kaufmann");
    result.setValue(Field.ADDRESS, "San Francisco");
    
    return result;
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  @Override
  protected String defaultClassifierString() {
    
    return "weka.classifiers.trees.DecisionStump";
  }

  /**
   * Select only instances with weights that contribute to 
   * the specified quantile of the weight distribution
   *
   * @param data the input instances
   * @param quantile the specified quantile eg 0.9 to select 
   * 90% of the weight mass
   * @return the selected instances
   */
  protected Instances selectWeightQuantile(Instances data, double quantile) { 

    int numInstances = data.numInstances();
    Instances trainData = new Instances(data, numInstances);
    double [] weights = new double [numInstances];

    double sumOfWeights = 0;
    for(int i = 0; i < numInstances; i++) {
      weights[i] = data.instance(i).weight();
      sumOfWeights += weights[i];
    }
    double weightMassToSelect = sumOfWeights * quantile;
    int [] sortedIndices = Utils.sort(weights);

    // Select the instances
    sumOfWeights = 0;
    for(int i = numInstances - 1; i >= 0; i--) {
      Instance instance = (Instance)data.instance(sortedIndices[i]).copy();
      trainData.add(instance);
      sumOfWeights += weights[sortedIndices[i]];
      if ((sumOfWeights > weightMassToSelect) && 
	  (i > 0) && 
	  (weights[sortedIndices[i]] != weights[sortedIndices[i - 1]])) {
	break;
      }
    }
    if (m_Debug) {
      System.err.println("Selected " + trainData.numInstances()
			 + " out of " + numInstances);
    }
    return trainData;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration listOptions() {

    Vector newVector = new Vector();

    newVector.addElement(new Option(
	"\tPercentage of weight mass to base training on.\n"
	+"\t(default 100, reduce to around 90 speed up)",
	"P", 1, "-P <num>"));
    
    newVector.addElement(new Option(
	"\tUse resampling for boosting.",
	"Q", 0, "-Q"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    
    return newVector.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -P &lt;num&gt;
   *  Percentage of weight mass to base training on.
   *  (default 100, reduce to around 90 speed up)</pre>
   * 
   * <pre> -Q
   *  Use resampling for boosting.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.DecisionStump)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.DecisionStump:
   * </pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {

    String thresholdString = Utils.getOption('P', options);
    if (thresholdString.length() != 0) {
      setWeightThreshold(Integer.parseInt(thresholdString));
    } else {
      setWeightThreshold(100);
    }
      
    setUseResampling(Utils.getFlag('Q', options));

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result = new Vector();

    if (getUseResampling())
      result.add("-Q");

    result.add("-P");
    result.add("" + getWeightThreshold());
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);

    return (String[]) result.toArray(new String[result.size()]);
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String weightThresholdTipText() {
    return "Weight threshold for weight pruning.";
  }

  /**
   * Set weight threshold
   *
   * @param threshold the percentage of weight mass used for training
   */
  public void setWeightThreshold(int threshold) {

    m_WeightThreshold = threshold;
  }

  /**
   * Get the degree of weight thresholding
   *
   * @return the percentage of weight mass used for training
   */
  public int getWeightThreshold() {

    return m_WeightThreshold;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useResamplingTipText() {
    return "Whether resampling is used instead of reweighting.";
  }

  /**
   * Set resampling mode
   *
   * @param r true if resampling should be done
   */
  public void setUseResampling(boolean r) {

    m_UseResampling = r;
  }

  /**
   * Get whether resampling is turned on
   *
   * @return true if resampling output is on
   */
  public boolean getUseResampling() {

    return m_UseResampling;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    if (super.getCapabilities().handles(Capability.NOMINAL_CLASS))
      result.enable(Capability.NOMINAL_CLASS);
    if (super.getCapabilities().handles(Capability.BINARY_CLASS))
      result.enable(Capability.BINARY_CLASS);
    
    return result;
  }

  /**
   * Boosting method.
   *
   * @param data the training data to be used for generating the
   * boosted classifier.
   * @throws Exception if the classifier could not be built successfully
   */

  @Override
  public void buildClassifier(Instances data) throws Exception {

    super.buildClassifier(data);
    m_Debug = false;

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    // only class? -> build ZeroR model
    if (data.numAttributes() == 1) {
      System.err.println(
	  "Cannot build model (only class attribute present in data!), "
	  + "using ZeroR model instead!");
      m_ZeroR = new weka.classifiers.rules.ZeroR();
      m_ZeroR.buildClassifier(data);
      return;
    }
    else {
      m_ZeroR = null;
    }
    
    setSeed(m_Seed);
    
    m_NumClasses = data.numClasses();
    if ((!m_UseResampling) && 
      (m_Classifier instanceof WeightedInstancesHandler)) {
        buildClassifierWithWeights(data);
      } else {
        buildClassifierUsingResampling(data);
      }
  }

  /**
   * Boosting method. Boosts using resampling
   *
   * @param data the training data to be used for generating the
   * boosted classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  protected void buildClassifierUsingResampling(Instances data) 
    throws Exception {

    int resamplingIterations = 0;
    int numInstances = data.numInstances();
    Random random = new Random(m_Seed);
    Instances sample;

    /**
     * Array for storing the predictions of the generated classifiers
     */
    double[][] predictions = new double[numInstances][];

    /**
     * Array for storing the weight for each instance (first index) and each
     * instance class label (second index).
     */
    double[][] w = new double[numInstances][m_NumClasses];

    /**
     * Array for storing the aggregated weight for each instance.
     */
    double[] W = new double[numInstances];

    /**
     * The label weighting function.
     */
    double[][] q = new double[numInstances][m_NumClasses];

    /**
     * The distribution over the instances is represented by the weight attribute
     * of each instance.
     */

    // Create a copy of the data so that when the weights are diddled
    // with it doesn't mess up the weights for anyone else
    Instances training = new Instances(data, 0, numInstances);

    /**
     * Temporary set of data result of applied sampling process. It is used for
     * training the classifiers.
     */
    Instances trainData;

    /**
     * Create the data manager with default parameters
     */
//    dm = new MwmoteDataManager(training);
    dm = new MismoteDataManager(training);

    // Initialize data
    m_Betas = new double [m_Classifiers.length];
    m_NumIterationsPerformed = 0;

    // Initialize the weight vector
    for (int i = 0; i < numInstances; i++) {
      for (int j = 0; j < m_NumClasses; j++) {
        if (j == training.instance(i).classValue()) continue;
        w[i][j] = training.instance(i).weight()/(m_NumClasses - 1);
      }
    }

    // Do boostrap iterations
    for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length; 
      m_NumIterationsPerformed++) {

      //Compute aggregated weights
      double totalW = 0;
      for (int i = 0; i < numInstances; i++) {
        W[i] = 0;
        for (int j = 0; j < m_NumClasses; j++) {
          W[i] += w[i][j];
        }
        totalW += W[i];
      }

      // Compute label weighting function and distribution
      for (int i = 0; i < numInstances; i++) {
        for (int j = 0; j < m_NumClasses; j++) {
          if (j == training.instance(i).classValue()) continue;
          q[i][j] = w[i][j]/W[i];
        }
        training.instance(i).setWeight(W[i]/totalW); // Sets the new distribution
      }

      if (m_Debug) {
        System.err.println("Training classifier " + (m_NumIterationsPerformed + 1));
        System.err.println();
        System.err.println("w[i][j] = " + weka.core.Utils.arrayToString(w));
        System.err.println();
        System.err.println("W[i] = " + weka.core.Utils.arrayToString(W));
        System.err.println();
        System.err.println("totalW = " + totalW);
        System.err.println();
        System.err.println("q[i][j] = " + weka.core.Utils.arrayToString(q));
        System.err.println();
        System.err.println(String.format("training.size() = %d", training.size()));
        System.err.println();
        System.err.print("training.instance(i).Weight() = [");
        for (int i = 0; i < training.size(); i++) {
          System.err.print(training.instance(i).weight() + ", ");
        }
        System.err.println("]");
        System.err.println();
      }

      /**
       * *** SAMPLING METHOD ***
       */

      // Determine randomly the number of negative instances after the pre-processing
      double x;
      int numNegAfterProcessing;
      int numPosAfterProcessing;
      if (beta == 0) {
        int QN;
        if (dm.negCount() < eta) {
          QN = dm.negCount();
        } else if (dm.posCount() >= eta) {
          QN = dm.posCount();
        } else {
          QN = eta;
        }
        int QP = (int)(QN * gamma / (100 - gamma));
        x = random.nextGaussian();
        numNegAfterProcessing = (int) (QN * (1.0 + sigma * x / 100.0));
        x = random.nextGaussian();
        numPosAfterProcessing = (int) (QP * (1.0 + sigma * x / 100.0));
      } else {
        x = random.nextGaussian();
        double Beta = beta + sigma * x;             // OJO: HICE UN CAMBIO EN ESTA FÓRMULA
        numNegAfterProcessing = (int) (dm.negCount() * Beta / 100.0);
        x = random.nextGaussian();
        double Gamma = gamma + sigma * x;
        numPosAfterProcessing = (int)(numNegAfterProcessing * Gamma / (100 - Gamma));
      }

      if (numNegAfterProcessing > dm.negCount() && !negBootstrap )
          numNegAfterProcessing = dm.negCount();

      ArrayList<Integer> DNIndexes;
      ArrayList<Integer> DPIndexes;

      if (m_Debug) {
        String msg = "Initial IR = %f    Final IR = %f    numPos = %d    numNeg = %d"
                + "    numPosAfterProcessing = %d    numNegAfterProcessing = %d";
        System.out.println(String.format(msg, (double)dm.negCount()/dm.posCount(),
                (numNegAfterProcessing > numPosAfterProcessing)?
                  (double)numNegAfterProcessing/numPosAfterProcessing:
                  (double)numPosAfterProcessing/numNegAfterProcessing,
                  dm.posCount(), dm.negCount(), numPosAfterProcessing, numNegAfterProcessing));
      }

      if (posBootstrap) {
        DPIndexes = bootstrapIndexes(random, dm.posCount(), dm.positives());
      } else {
        DPIndexes = copyIndexes(dm.positives());
      }

      //Instances synthetics = mismote(numPosAfterProcessing - dm.posCount(), training);
      //Instances synthetics = mwmote4mil(numPosAfterProcessing - dm.posCount(), training);
      Instances synthetics = bagsmote(numPosAfterProcessing - dm.posCount(), training);

      if (negBootstrap) {
        DNIndexes = bootstrapIndexes(random, numNegAfterProcessing, dm.negatives());
      } else {
        DNIndexes = subsampleIndexes(numNegAfterProcessing, dm.negatives());
      }

      /**
       * DPIndexes union DNIndexes trying to keep the same order as in training
       */
      ArrayList<Integer> subsampledIndexes = new ArrayList<Integer>(training.size());
      for (int i = 0; i < training.size(); i++) {
        subsampledIndexes.add(null);
      }
      for (int i = 0; i < DNIndexes.size(); i++) {
        int index = DNIndexes.get(i).intValue();
        subsampledIndexes.set(index, DNIndexes.get(i));
      }
      for (int i = 0; i < DPIndexes.size(); i++) {
        int index = DPIndexes.get(i).intValue();
        subsampledIndexes.set(index, DPIndexes.get(i));
      }

      double sumOfWeights = 0;
      int subsampledCount = 0;
      for (int i = 0; i < subsampledIndexes.size(); i++) {
        if (subsampledIndexes.get(i) == null) continue;
        int index = subsampledIndexes.get(i).intValue();
        sumOfWeights += W[index];
        subsampledCount++;
      }

      int newSize = subsampledCount + (synthetics.size() - training.size());
      double normFactor = (double) subsampledCount / (sumOfWeights * newSize);
      trainData = new Instances(synthetics, newSize);
      for (int i = 0; i < subsampledIndexes.size(); i++) {
        if (subsampledIndexes.get(i) == null) continue;
        int index = subsampledIndexes.get(i).intValue();
        Instance X = (Instance)training.get(index).copy();
        X.setWeight(W[index] * normFactor);
        trainData.add(X);
      }
      for (int i = training.size(); i < synthetics.size(); i++) {
        Instance X = synthetics.get(i);
        X.setWeight(1.0 / newSize);
        trainData.add(X);
      }

      // Select instances to train the classifier on
      /*
      if (m_WeightThreshold < 100) {
        trainData = selectWeightQuantile(training, (double)m_WeightThreshold / 100);
      } else {
        trainData = new Instances(training);
      }
      double sampledWeight = trainData.sumOfWeights();
      */

      if (m_Debug) {
        double sampledWeight = trainData.sumOfWeights();
        System.err.println("****** Normalize weights of sampled data ******");
        System.err.println();
        System.err.println(String.format("trainData.size() = %d", trainData.size()));
        System.err.println();
        System.err.println(String.format("trainData.sumOfWeights() = %f", sampledWeight));
        System.err.println();
        System.err.print("trainData.instance(i).Weight() = [");
        for (int i = 0; i < trainData.size(); i++) {
          System.err.print(trainData.instance(i).weight() + ", ");
        }
        System.err.println("]");
        System.err.println();
      }


      // Resample
      resamplingIterations = 0;
      double[] weights = new double[trainData.numInstances()];
      for (int i = 0; i < trainData.numInstances(); i++) {
        weights[i] = trainData.instance(i).weight();
      }

      //double error;
      double epsilon;
      do {
        sample = trainData.resampleWithWeights(random, weights);

        // Build and evaluate classifier
        m_Classifiers[m_NumIterationsPerformed].buildClassifier(sample);

        //Evaluation evaluation = new Evaluation(data);
        // Calculate the pseudo-loss
        epsilon = 0;
        for (int i = 0; i < numInstances; i++) {
          predictions[i] = m_Classifiers[m_NumIterationsPerformed].distributionForInstance(training.instance(i));
          double errorWeight = 0;
          for (int j = 0; j < m_NumClasses; j++) {
            if (j == training.instance(i).classValue()) continue;
            errorWeight += q[i][j] * predictions[i][j];
          }
          epsilon += training.instance(i).weight()
                  * (1 - predictions[i][(int)training.instance(i).classValue()] + errorWeight);
        }
        epsilon *= 0.5;
        if (m_Debug) {
          System.err.println();
          System.err.println(String.format("epsilon = %f", epsilon));
        }
        //error = evaluation.errorRate();
        resamplingIterations++;
      } while (Utils.eq(epsilon, 0) &&      //  is it epsilon or error ????
	      (resamplingIterations < MAX_NUM_RESAMPLING_ITERATIONS));

      if (m_Debug) {
        System.err.println();
        System.err.println("Iterations finished");
        System.err.println();
      }
      /*
      // Stop if error too big or 0
      if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0)) {
        if (m_NumIterationsPerformed == 0) {
          m_NumIterationsPerformed = 1; // If we're the first we have to to use it
        }
        break;
      }
       *
       */
      
      // Determine the weight to assign to this model
      m_Betas[m_NumIterationsPerformed] = Math.log((1 - epsilon) / epsilon);
      double reweight = epsilon / (1 - epsilon);
      //double reweight = (1 - epsilon) / epsilon;
      if (m_Debug) {
        System.err.println("\terror rate = " + epsilon
			   +"  beta = " + m_Betas[m_NumIterationsPerformed]);
        System.err.println(String.format("reweight = %f", reweight));
        System.err.println();
      }
 
      // Set the new weights vector
      for (int i = 0; i < numInstances; i++) {
        for (int j = 0; j < m_NumClasses; j++) {
          if (j == training.instance(i).classValue()) continue;
          double exp = 0.5 * (1 + predictions[i][(int)training.instance(i).classValue()] -
                  predictions[i][j]);
          w[i][j] *= Math.pow(reweight, exp);
        }
      }
    }
    if (m_Debug) {
      System.err.println();
      System.err.println("****  Model finished *****");
      System.err.println();
    }
  }

    /**
     * Apply random sampling with replacement (bootstrapping) to a dataset
     * returning a list of indexes of selected instances.
     *
     * @param n number of instance to be sampled
     * @param D set of instances from which to sample
     * @return index list of sampled instances
     */
    private ArrayList<Integer> bootstrapIndexes(Random random, int n, ArrayList<Integer> D) {
        // New empty set of indexes
        ArrayList<Integer> newDataIndexes = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) {
          int r = random.nextInt(D.size());
          newDataIndexes.add(D.get(r));
        }
        return newDataIndexes;
    }

    /**
     * Apply a copy to a dataset returning a list of indexes of copied instances.
     *
     * @param D set of instances to be copied
     * @return index list of copied instances
     */
    private ArrayList<Integer> copyIndexes(ArrayList<Integer> DIndexes) {
      return (ArrayList<Integer>)DIndexes.clone();
    }

    /**
     * Apply random sampling without replacement to a dataset returning a list
     * of indexes of selected instances.
     *
     * @param n number of instance to be sampled
     * @param DIndexes set of instances from which to sample
     * @return index list of sampled instances
     * @throws IllegalArgumentException if the  number of instances to be sampled
     * is larger than the size of the set of instances from which sampling is performed.
     *
     */
    private ArrayList<Integer> subsampleIndexes(int n, ArrayList<Integer> DIndexes) throws IllegalArgumentException {
        // Initialize the set of new indexes to the indexes for the supplied dataset.
        ArrayList<Integer> newDataIndexes = (ArrayList<Integer> )DIndexes.clone();
        if (n > DIndexes.size()) {
            throw new IllegalArgumentException("The number of instances to be sampled can not be "
                    + "larger than the size of the set of instances from which sampling is performed:"
                    + " n = " + n + " D.size = " + DIndexes.size());
        } else {
          // Then remove indexes randomly
          Random random = new Random(m_Seed);
          for (int i = 0; i < DIndexes.size() - n; i++) {
            int r = random.nextInt(newDataIndexes.size());
            newDataIndexes.remove(r);
          }
        }
        return newDataIndexes;
    }

    private Instances bagsmote(int n, Instances D) throws Exception {
      BagSMOTE bagsmote = new BagSMOTE();
      bagsmote.setInputFormat(D);
      bagsmote.setNumSynthetics(n);
      Instances syn = Filter.useFilter(D, bagsmote);
      return syn;
    }

//    private Instances mismote(int n, Instances D) throws Exception {
//      MISMOTE mismote = new MISMOTE();
//      mismote.setInputFormat(D);
//      mismote.SetNumSynthetics(n);
//      mismote.setDataManager(dm);
//      Instances syn = Filter.useFilter(D, mismote);
//      return syn;
//    }
//
//    private Instances mwmote4mil(int n, Instances D) throws Exception {
//      MWMOTE4MIL mwmote4mil = new MWMOTE4MIL();
//      mwmote4mil.setInputFormat(D);
//      mwmote4mil.setNumSynthetics(n);
//      mwmote4mil.setDataManager(dm);
//      Instances syn = Filter.useFilter(D, mwmote4mil);
//      return syn;
//    }
//
  /**
   * Boosting method. Boosts any classifier that can handle weighted
   * instances.
   *
   * @param data the training data to be used for generating the
   * boosted classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  protected void buildClassifierWithWeights(Instances data) 
    throws Exception {

    int numInstances = data.numInstances();
    Random random = new Random(m_Seed);

    /**
     * Array for storing the predictions of the generated classifiers
     */
    double[][] predictions = new double[numInstances][];

    /**
     * Array for storing the weight for each instance (first index) and each
     * instance class label (second index).
     */
    double[][] w = new double[numInstances][m_NumClasses];

    /**
     * Array for storing the aggregated weight for each instance.
     */
    double[] W = new double[numInstances];

    /**
     * The label weighting function.
     */
    double[][] q = new double[numInstances][m_NumClasses];

    /**
     * The distribution over the instances is represented by the weight attribute
     * of each instance.
     */

    // Create a copy of the data so that when the weights are diddled
    // with it doesn't mess up the weights for anyone else
    Instances training = new Instances(data, 0, numInstances);

    /**
     * Temporary set of data result of applied sampling process. It is used for
     * training the classifiers.
     */
    Instances trainData;

    // Find the posive class label
    AttributeStats classStats = data.attributeStats(data.classIndex());
    int posLabel = Utils.minIndex(classStats.nominalCounts);

    // Create empty sets for storing the negative and positive instance index
    ArrayList<Integer> negatives = new ArrayList<Integer>();
    ArrayList<Integer> positives = new ArrayList<Integer>();

    // Adds the positive and negative instance indexes to the respective sets
    for (int i = 0; i < training.numInstances(); i++) {
        if (training.get(i).classValue() == posLabel) {
            positives.add(i);
        } else {
            negatives.add(i);
        }
    }

    // Record the number of positive and negative instances
    int numPos = positives.size();
    int numNeg = negatives.size();

      // Initialize data
    m_Betas = new double [m_Classifiers.length];
    m_NumIterationsPerformed = 0;

    // Initialize the weight vector
    for (int i = 0; i < numInstances; i++) {
      for (int j = 0; j < m_NumClasses; j++) {
        if (j == training.instance(i).classValue()) continue;
        w[i][j] = training.instance(i).weight()/(m_NumClasses - 1);
      }
    }

    /**
     * ***  Do boostrap iterations  ***
     */
    for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length; 
      m_NumIterationsPerformed++) {

      //Compute aggregated weights
      double totalW = 0;
      for (int i = 0; i < numInstances; i++) {
        W[i] = 0;
        for (int j = 0; j < m_NumClasses; j++) {
          W[i] += w[i][j];
        }
        totalW += W[i];
      }

      // Compute label weighting function and distribution
      for (int i = 0; i < numInstances; i++) {
        for (int j = 0; j < m_NumClasses; j++) {
          if (j == training.instance(i).classValue()) continue;
          q[i][j] = w[i][j]/W[i];
        }
        training.instance(i).setWeight(W[i]/totalW); // Sets the new distribution
      }

      if (m_Debug) {
        System.err.println("Training classifier " + (m_NumIterationsPerformed + 1));
        System.err.println();
        System.err.println("w[i][j] = " + weka.core.Utils.arrayToString(w));
        System.err.println();
        System.err.println("W[i] = " + weka.core.Utils.arrayToString(W));
        System.err.println();
        System.err.println("totalW = " + totalW);
        System.err.println();
        System.err.println("q[i][j] = " + weka.core.Utils.arrayToString(q));
        System.err.println();
        System.err.println(String.format("training.size() = %d", training.size()));
        System.err.println();
        System.err.print("training.instance(i).Weight() = [");
        for (int i = 0; i < training.size(); i++) {
          System.err.print(training.instance(i).weight() + ", ");
        }
        System.err.println("]");
        System.err.println();
      }

      /**
       * *** SAMPLING METHOD ***
       */

      // Determine randomly the number of negative instances after the pre-processing
      double x = Math.abs(random.nextGaussian());
      double Gamma = gamma - sigma * x;             // OJO: HICE UN CAMBIO EN ESTA FÓRMULA
      //int numNegAfterProcessing = (int) (numPos * (100 / Gamma - 1));
      int numPosAfterProcessing = (int)(numNeg * Gamma / (100 - Gamma));

      ArrayList<Integer> DNIndexes = null;
      ArrayList<Integer> DPIndexes = null;

      if (m_Debug) {
        String msg = "Initial IR = %f    Final IR = %f    numPosAfterProcessing = %d    numNeg = %d";
        System.out.println(String.format(msg, (double)numNeg/numPos,
                (numNeg > numPosAfterProcessing)? (double)numNeg/numPosAfterProcessing:
                  (double)numPosAfterProcessing/numNeg, numPosAfterProcessing, numNeg));
      }

      if (negBootstrap) {
        DNIndexes = bootstrapIndexes(random, numNeg, negatives);
      } else {
        DNIndexes = copyIndexes(negatives);
      }
      //DNIndexes = resampleWithWeights(negatives, random);

      DPIndexes = copyIndexes(positives);
      Instances synthetics = bagsmote(numPosAfterProcessing - dm.posCount(), training);
      //Instances synthetics = mwmote4mil(numPosAfterProcessing - dm.posCount(), training);
//      Instances synthetics = mismote(numPosAfterProcessing - numPos, training);

      /**
       * DPIndexes union DNIndexes trying to keep the same order as in training
       */
      ArrayList<Integer> subsampledIndexes = new ArrayList<Integer>(training.size());
      for (int i = 0; i < training.size(); i++) {
        subsampledIndexes.add(null);
      }
      for (int i = 0; i < DNIndexes.size(); i++) {
        int index = DNIndexes.get(i).intValue();
        subsampledIndexes.set(index, DNIndexes.get(i));
      }
      for (int i = 0; i < DPIndexes.size(); i++) {
        int index = DPIndexes.get(i).intValue();
        subsampledIndexes.set(index, DPIndexes.get(i));
      }

      //double sumOfWeights = 0;
      int subsampledCount = 0;
      for (int i = 0; i < subsampledIndexes.size(); i++) {
        if (subsampledIndexes.get(i) == null) continue;
        //int index = subsampledIndexes.get(i).intValue();
        //sumOfWeights += W[index];
        subsampledCount++;
      }

      int newSize = subsampledCount + (synthetics.size() - training.size());
      double normFactor = (double) training.size() / newSize;
      trainData = new Instances(synthetics, newSize);
      for (int i = 0; i < subsampledIndexes.size(); i++) {
        if (subsampledIndexes.get(i) == null) continue;
        int index = subsampledIndexes.get(i).intValue();
        Instance X = (Instance)training.get(index).copy();
        X.setWeight(W[index] * normFactor);
        trainData.add(X);
      }
      for (int i = training.size(); i < synthetics.size(); i++) {
        Instance X = synthetics.get(i);
        X.setWeight(1.0 / newSize);
        trainData.add(X);
      }

      // Select instances to train the classifier on
      /*
      if (m_WeightThreshold < 100) {
        trainData = selectWeightQuantile(training, (double)m_WeightThreshold / 100);
      } else {
        trainData = new Instances(training);
      }
      double sampledWeight = trainData.sumOfWeights();
      */

      // Build the classifier
      if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable)
        ((Randomizable) m_Classifiers[m_NumIterationsPerformed]).setSeed(random.nextInt());
      m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainData);

      // Calculate the pseudo-loss
      double epsilon = 0;
      for (int i = 0; i < numInstances; i++) {
        predictions[i] = m_Classifiers[m_NumIterationsPerformed].distributionForInstance(training.instance(i));
        double errorWeight = 0;
        for (int j = 0; j < m_NumClasses; j++) {
          if (j == training.instance(i).classValue()) continue;
          errorWeight += q[i][j] * predictions[i][j];
        }
        epsilon += training.instance(i).weight()
                * (1 - predictions[i][(int)training.instance(i).classValue()] + errorWeight);
      }
      epsilon *= 0.5;

      /*
      // Stop if error too small or error too big and ignore this model
      if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0)) {
        if (m_NumIterationsPerformed == 0) {
          m_NumIterationsPerformed = 1; // If we're the first we have to to use it
        }
        break;
      }
       */

      // Determine the weight to assign to this model
      m_Betas[m_NumIterationsPerformed] = Math.log((1 - epsilon) / epsilon);
      double reweight = epsilon / (1 - epsilon);
      if (m_Debug) {
        System.err.println("\terror rate = " + epsilon
			   +"  beta = " + m_Betas[m_NumIterationsPerformed]);
      }
 
      // Set the new weights vector
      for (int i = 0; i < numInstances; i++) {
        for (int j = 0; j < m_NumClasses; j++) {
          if (j == training.instance(i).classValue()) continue;
          double exp = 0.5 * (1 + predictions[i][(int)training.instance(i).classValue()] -
                  predictions[i][j]);
          w[i][j] *= Math.pow(reweight, exp);
        }
      }
    }
  }
  
  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @throws Exception if instance could not be classified
   * successfully
   */
  @Override
  public double [] distributionForInstance(Instance instance) 
    throws Exception {
      
    // default model?
    if (m_ZeroR != null) {
      return m_ZeroR.distributionForInstance(instance);
    }
    
    if (m_NumIterationsPerformed == 0) {
      throw new Exception("No model built");
    }
    
    if (m_NumIterationsPerformed == 1) {
      return m_Classifiers[0].distributionForInstance(instance);
    } else {
      double [] sums = new double [instance.numClasses()];
      double [] dist = new double [instance.numClasses()];
      for (int i = 0; i < m_NumIterationsPerformed; i++) {
        dist = m_Classifiers[i].distributionForInstance(instance);
        for (int k = 0; k < m_NumClasses; k++) {
          sums[k] += m_Betas[i] * dist[k];
        }
      }
      return Utils.logs2probs(sums);
    }
  }

  /**
   * Returns the boosted model as Java source code.
   *
   * @param className the classname of the generated class
   * @return the tree as Java source code
   * @throws Exception if something goes wrong
   */
  public String toSource(String className) throws Exception {

    if (m_NumIterationsPerformed == 0) {
      throw new Exception("No model built yet");
    }
    if (!(m_Classifiers[0] instanceof Sourcable)) {
      throw new Exception("Base learner " + m_Classifier.getClass().getName()
			  + " is not Sourcable");
    }

    StringBuilder text = new StringBuilder("class ");
    text.append(className).append(" {\n\n");

    text.append("  public static double classify(Object[] i) {\n");

    if (m_NumIterationsPerformed == 1) {
      text.append("    return ").append(className).append("_0.classify(i);\n");
    } else {
      text.append("    double [] sums = new double [").append(m_NumClasses).append("];\n");
      for (int i = 0; i < m_NumIterationsPerformed; i++) {
	      text.append("    sums[(int) ").append(className).append('_').append(i).append(".classify(i)] += ").append(m_Betas[i]).append(";\n");
      }
      text.append("    double maxV = sums[0];\n" + "    int maxI = 0;\n" + "    for (int j = 1; j < ").append(m_NumClasses).append("; j++) {\n"+
		  "      if (sums[j] > maxV) { maxV = sums[j]; maxI = j; }\n"+
		  "    }\n    return (double) maxI;\n");
    }
    text.append("  }\n}\n");

    for (int i = 0; i < m_Classifiers.length; i++) {
	text.append(((Sourcable)m_Classifiers[i])
		    .toSource(className + '_' + i));
    }
    return text.toString();
  }

  /**
   * Returns description of the boosted classifier.
   *
   * @return description of the boosted classifier as a string
   */
  @Override
  public String toString() {
    
    // only ZeroR model?
    if (m_ZeroR != null) {
      StringBuilder buf = new StringBuilder();
      buf.append(this.getClass().getName().replaceAll(".*\\.", "")).append("\n");
      buf.append(this.getClass().getName().replaceAll(".*\\.", "").replaceAll(".", "=")).append("\n\n");
      buf.append("Warning: No model could be built, hence ZeroR model is used:\n\n");
      buf.append(m_ZeroR.toString());
      return buf.toString();
    }
    
    StringBuilder text = new StringBuilder();
    
    if (m_NumIterationsPerformed == 0) {
      text.append("AdaBoostM1: No model built yet.\n");
    } else if (m_NumIterationsPerformed == 1) {
      text.append("AdaBoostM1: No boosting possible, one classifier used!\n");
      text.append(m_Classifiers[0].toString()).append("\n");
    } else {
      text.append("AdaBoostM1: Base classifiers and their weights: \n\n");
      for (int i = 0; i < m_NumIterationsPerformed ; i++) {
	      text.append(m_Classifiers[i].toString()).append("\n\n");
	      text.append("Weight: ").append(Utils.roundDouble(m_Betas[i], 2)).append("\n\n");
      }
      text.append("Number of performed Iterations: ").append(m_NumIterationsPerformed).append("\n");
    }
    
    return text.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8034 $");
  }

  public int getSigma() {
      return sigma;
  }

  public void setSigma(int sigma) {
    if (sigma < 0)
      throw new IllegalArgumentException("Sigma should be a non negative number");
    this.sigma = sigma;
  }

  public double getGamma() {
      return gamma;
  }

  public void setGamma(int gamma) {
      this.gamma = gamma;
  }

  public boolean isNegBoostrap() {
      return negBootstrap;
  }

  public void setNegBoostrap(boolean negBoostrap) {
      this.negBootstrap = negBoostrap;
  }

  public boolean isPosBootstrap() {
      return posBootstrap;
  }

  public void setPosBootstrap(boolean posBootstrap) {
      this.posBootstrap = posBootstrap;
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
    runClassifier(new MISORUBoostBS(), argv);
  }

  /**
   * @return the eta
   */
  public int getEta() {
    return eta;
  }

  /**
   * @param eta the eta to set
   */
  public void setEta(int eta) {
    this.eta = eta;
  }
}
