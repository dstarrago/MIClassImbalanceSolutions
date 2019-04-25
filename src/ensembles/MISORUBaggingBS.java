/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ensembles;

import dataPreprocessing.BagSMOTE;
import dataPreprocessing.MISMOTE;
import dataPreprocessing.MWMOTE4MIL;
import dataPreprocessing.MismoteDataManager;
import dataPreprocessing.MwmoteDataManager;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.core.AdditionalMeasureProducer;
import weka.core.AttributeStats;
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

public class MISORUBaggingBS
        extends RandomizableParallelIteratedSingleClassifierEnhancer
        implements WeightedInstancesHandler, AdditionalMeasureProducer,
        TechnicalInformationHandler {

  /**
   * Per cent of the total desired training examples to be represented by the minority class
   */
  private int gamma = 50;

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

//  private ArrayList<Integer> negatives;
//  private ArrayList<Integer> positives;
  private int numPosIn;
  private int numNegIn;
  private int numPosOut;
  private int numNegOut;
  protected Random random;
  protected boolean[][] m_inBag;
  protected Instances m_data;
  //private MwmoteDataManager dm;
  private MismoteDataManager dm;


    //utilizando la clase MISMOTE del proyecto MILPro3
//    private MISMOTE mismote;

    public int getBeta() {
        return beta;
    }

    public void setBeta(int beta) {
        this.beta = beta;
    }

    public int getN1() {
        return numNegOut;
    }

    public void setN1(int n1) {
        this.numNegOut = n1;
    }

    public int getSigma() {
        return sigma;
    }

    public void setSigma(int sigma) {
        this.sigma = sigma;
    }

    public int getGamma() {
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
     * for serialization
     */
    static final long serialVersionUID = -505879962237199703L;

    /**
     * The size of each bag sample, as a percentage of the training size
     */
    protected int m_BagSizePercent = 100;

    /**
     * Whether to calculate the out of bag error
     */
    protected boolean m_CalcOutOfBag = false;

    /**
     * The out of bag error that has been calculated
     */
    protected double m_OutOfBagError;

    /**
     * Constructor.
     */
    public MISORUBaggingBS() {

        m_Classifier = new weka.classifiers.mi.MITI();
//        mismote = new MISMOTE();

    }

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {

        return "Class for bagging a classifier to reduce variance. Can do classification "
                + "and regression depending on the base learner. \n\n"
                + "For more information, see\n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Leo Breiman");
        result.setValue(Field.YEAR, "1996");
        result.setValue(Field.TITLE, "Bagging predictors");
        result.setValue(Field.JOURNAL, "Machine Learning");
        result.setValue(Field.VOLUME, "24");
        result.setValue(Field.NUMBER, "2");
        result.setValue(Field.PAGES, "123-140");

        return result;
    }

    /**
     * String describing default classifier.
     *
     * @return the default classifier classname
     */
    protected String defaultClassifierString() {

        return "weka.classifiers.trees.REPTree";
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(2);

        newVector.addElement(new Option(
                "\tSize of each bag, as a percentage of the\n"
                + "\ttraining set size. (default 100)",
                "P", 1, "-P"));
        newVector.addElement(new Option(
                "\tCalculate the out of bag error.",
                "O", 0, "-O"));

        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        return newVector.elements();
    }


    /*
     * Options after -- are passed to the designated classifier.<numPosIn>
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String bagSize = Utils.getOption('P', options);
        if (bagSize.length() != 0) {
            setBagSizePercent(Integer.parseInt(bagSize));
        } else {
            setBagSizePercent(100);
        }

        setCalcOutOfBag(Utils.getFlag('O', options));

        super.setOptions(options);
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {

        String[] superOptions = super.getOptions();
        String[] options = new String[superOptions.length + 3];

        int current = 0;
        options[current++] = "-P";
        options[current++] = "" + getBagSizePercent();

        if (getCalcOutOfBag()) {
            options[current++] = "-O";
        }

        System.arraycopy(superOptions, 0, options, current,
                superOptions.length);

        current += superOptions.length;
        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String bagSizePercentTipText() {
        return "Size of each bag, as a percentage of the training set size.";
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public int getBagSizePercent() {

        return m_BagSizePercent;
    }

    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newBagSizePercent the bag size, as a percentage.
     */
    public void setBagSizePercent(int newBagSizePercent) {

        m_BagSizePercent = newBagSizePercent;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String calcOutOfBagTipText() {
        return "Whether the out-of-bag error is calculated.";
    }

    /**
     * Set whether the out of bag error is calculated.
     *
     * @param calcOutOfBag whether to calculate the out of bag error
     */
    public void setCalcOutOfBag(boolean calcOutOfBag) {

        m_CalcOutOfBag = calcOutOfBag;
    }

    /**
     * Get whether the out of bag error is calculated.
     *
     * @return whether the out of bag error is calculated
     */
    public boolean getCalcOutOfBag() {

        return m_CalcOutOfBag;
    }

    /**
     * Gets the out of bag error that was calculated as the classifier was
     * built.
     *
     * @return the out of bag error
     */
    public double measureOutOfBagError() {

        return m_OutOfBagError;
    }

    /**
     * Returns an enumeration of the additional measure names.
     *
     * @return an enumeration of the measure names
     */
    public Enumeration enumerateMeasures() {

        Vector newVector = new Vector(1);
        newVector.addElement("measureOutOfBagError");
        return newVector.elements();
    }

    /**
     * Returns the value of the named measure.
     *
     * @param additionalMeasureName the name of the measure to query for its
     * value
     * @return the value of the named measure
     * @throws IllegalArgumentException if the named measure is not supported
     */
    public double getMeasure(String additionalMeasureName) {

        if (additionalMeasureName.equalsIgnoreCase("measureOutOfBagError")) {
            return measureOutOfBagError();
        } else {
            throw new IllegalArgumentException(additionalMeasureName
                    + " not supported (Bagging)");
        }
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

    /**
     * Creates a new dataset of the same size using random sampling with
     * replacement according to the given weight vector. The weights of the
     * instances in the new dataset are set to one. The length of the weight
     * vector has to be the same as the number of instances in the dataset, and
     * all weights have to be positive.
     *
     * @param data the data to be sampled from
     * @param random a random number generator
     * @param sampled indicating which instance has been sampled
     * @return the new dataset
     * @throws IllegalArgumentException if the weights array is of the wrong
     * length or contains negative weights.
     */
    public final Instances resampleWithWeights(Instances data,
            Random random,
            boolean[] sampled) {

        double[] weights = new double[data.numInstances()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = data.instance(i).weight();
        }
        Instances newData = new Instances(data, data.numInstances());
        if (data.numInstances() == 0) {
            return newData;
        }
        double[] probabilities = new double[data.numInstances()];
        double sumProbs = 0, sumOfWeights = Utils.sum(weights);
        for (int i = 0; i < data.numInstances(); i++) {
            sumProbs += random.nextDouble();
            probabilities[i] = sumProbs;
        }
        Utils.normalize(probabilities, sumProbs / sumOfWeights);

        // Make sure that rounding errors don't mess things up
        probabilities[data.numInstances() - 1] = sumOfWeights;
        int k = 0;
        int l = 0;
        sumProbs = 0;
        while ((k < data.numInstances() && (l < data.numInstances()))) {
            if (weights[l] < 0) {
                throw new IllegalArgumentException("Weights have to be positive.");
            }
            sumProbs += weights[l];
            while ((k < data.numInstances())
                    && (probabilities[k] <= sumProbs)) {
                newData.add(data.instance(l));
                sampled[l] = true;
                newData.instance(k).setWeight(1);
                k++;
            }
            l++;
        }
        return newData;
    }

   /**
     * Devuelve un cjto de instancias remuestreadas por bootstrap
     *
     * @param numNegIn numero aleatorio de instancias
     * @param D cjto de instancias a remuestrear
     * @return el cjto de instancias remuestreadas
     */
    private Instances bootstrap(int n, Instances D) {
        //se crea el cjto nuevo de instancias vacio
        Instances newData = new Instances(D, n);
        Random rand=new Random(m_Seed);
        for (int i = 0; i < n; i++) {
            //generar el num aleatorio entre 1 y la cant de instancias
            int r = rand.nextInt(D.numInstances());
            //añadir al nuevo cjto de instancias la instancia D[r]
            newData.add(D.get(r));
        }
        return newData;
    }

    /**
     * Devuelve un cjto de instancias remuestreadas sin reemplazamiento
     *
     * @param numNegIn numero aleatorio de instancias
     * @param D cjto de instancias a remuestrear
     * @return el cjto de instancias remuestreadas
     */
    //metodo para el submuestro sin reemplazo
    private Instances submuestrear(int n, Instances D) throws IllegalArgumentException {
        //nuevo set de instancias
        Instances newData = new Instances(D, n);
        //copia de D
        Random rand=new Random(m_Seed);
        Instances pool = new Instances(D);
        if (n >= D.numInstances()) {
            throw new IllegalArgumentException("El numero de instancias a muestrear no puede ser mayor"
                    + " que el tamaño del conjunto de instancias desde el cuál se realiza el muestreo!!!");
        } else {
            //generar un numero aleatorio entre 1 y D.numInstances
            int r = rand.nextInt(pool.numInstances());
            Instance X = pool.get(r);
            newData.add(X);
            //eliminar X de pool para hacer el submuestreo sin reemplazamiento
            pool.remove(r);
        }
        return newData;
        
    }

    private void determineOutputSize() {
      double x;
      if (beta == 0) {
        int QN;
        if (numNegIn < eta) {
          QN = numNegIn;
        } else if (numPosIn >= eta) {
          QN = numPosIn;
        } else {
          QN = eta;
        }
        int QP = (int)(QN * gamma / (100 - gamma));
        x = random.nextGaussian();
        numNegOut = (int) (QN * (1.0 + sigma * x / 100.0));
        x = random.nextGaussian();
        numPosOut = (int) (QP * (1.0 + sigma * x / 100.0));
      } else {
        x = random.nextGaussian();
        double Beta = beta + sigma * x;             // OJO: HICE UN CAMBIO EN ESTA FÓRMULA
        numNegOut = (int) (numNegIn * Beta / 100.0);
        x = random.nextGaussian();
        double Gamma = gamma + sigma * x;
        numPosOut = (int)(numNegOut * Gamma / (100 - Gamma));
      }
      if (numNegOut > numNegIn && !negBootstrap )
          numNegOut = numNegIn;
      if (m_Debug) {
        String msg = "Initial IR = %f    Final IR = %f    numPos = %d    numNeg = %d"
                + "    numPosAfterProcessing = %d    numNegAfterProcessing = %d";
        System.out.println(String.format(msg, (double)numNegIn/numPosIn,
                (numNegOut > numPosOut)? (double)numNegOut/numPosOut:
                  (double)numPosOut/numNegOut,
                  numPosIn, numNegIn, numPosOut, numNegOut));
      }
    }

    /**
     * @param iteration the number of the iteration for the requested training
     * set.
     * @return the training set for the supplied iteration number
     * @throws Exception if something goes wrong when generating a training set.
     */
    protected synchronized Instances getTrainingSet(int iteration) throws Exception {
      determineOutputSize();

      ArrayList<Integer> DNIndexes;
      ArrayList<Integer> DPIndexes;

      if (posBootstrap) {
        DPIndexes = bootstrapIndexes(random, numPosIn, dm.positives());
      } else {
        DPIndexes = copyIndexes(dm.positives());
      }

      //Instances synthetics = mismote(numPosOut - numPosIn, m_data);
      //Instances synthetics = mwmote4mil(numPosOut - numPosIn, m_data);
      Instances synthetics = bagsmote(numPosOut - numPosIn, m_data);

      if (negBootstrap) {
        DNIndexes = bootstrapIndexes(random, numNegOut, dm.negatives());
      } else {
        DNIndexes = subsampleIndexes(numNegOut, dm.negatives());
      }

      /**
       * DPIndexes union DNIndexes trying to keep the same order as in training
       */
      ArrayList<Integer> subsampledIndexes = new ArrayList<Integer>(m_data.size());
      for (int i = 0; i < m_data.size(); i++) {
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

      int subsampledCount = 0;
      for (int i = 0; i < subsampledIndexes.size(); i++) {
        if (subsampledIndexes.get(i) != null)
          subsampledCount++;
      }

      int newSize = subsampledCount + (synthetics.size() - m_data.size());
      Instances bagData = new Instances(synthetics, newSize);
      for (int i = 0; i < subsampledIndexes.size(); i++) {
        if (subsampledIndexes.get(i) == null) continue;
        int index = subsampledIndexes.get(i).intValue();
        Instance X = (Instance)m_data.get(index).copy();
        bagData.add(X);
      }
      for (int i = m_data.size(); i < synthetics.size(); i++) {
        Instance X = synthetics.get(i);
        //X.setWeight(1.0 / newSize);
        bagData.add(X);
      }

      /*Random r = new Random(m_Seed + iteration);

       // create the in-bag dataset
       if (m_CalcOutOfBag) {
       m_inBag[iteration] = new boolean[m_data.numInstances()];
       bagData = resampleWithWeights(m_data, r, m_inBag[iteration]);
       } else {
       bagData = m_data.resampleWithWeights(r);
       if (bagSize < m_data.numInstances()) {
       bagData.randomize(r);
       Instances newBagData = new Instances(bagData, 0, bagSize);
       bagData = newBagData;
       }
       }
       */
      return bagData;
    }

    /**
     * Bagging method.
     *
     * @param data the training data to be used for generating the bagged
     * classifier.
     * @throws Exception if the classifier could not be built successfully
     */
  @Override
    public void buildClassifier(Instances data) throws Exception {

      // can classifier handle the data?
      getCapabilities().testWithFail(data);

      // remove instances with missing class
      m_data = new Instances(data);
      m_data.deleteWithMissingClass();

      super.buildClassifier(m_data);

      random = new Random(m_Seed);

      if (m_CalcOutOfBag && (m_BagSizePercent != 100)) {
          throw new IllegalArgumentException("Bag size needs to be 100% if "
                  + "out-of-bag error is to be calculated!");
      }

      //int bagSize = m_data.numInstances() * m_BagSizePercent / 100;

      m_inBag = null;
      if (m_CalcOutOfBag) {
          m_inBag = new boolean[m_Classifiers.length][];
      }

      for (int j = 0; j < m_Classifiers.length; j++) {
          if (m_Classifier instanceof Randomizable) {
              ((Randomizable) m_Classifiers[j]).setSeed(random.nextInt());
          }
      }

    /**
     * Create the data manager with default parameters
     */
//    dm = new MwmoteDataManager(m_data);
      dm = new MismoteDataManager(m_data);

//      // Determinar la clase positiva
//      AttributeStats classStats = m_data.attributeStats(m_data.classIndex());
//      int posLabel = Utils.minIndex(classStats.nominalCounts);

//      positives = new ArrayList<Integer>(classStats.nominalCounts[posLabel]);
//      negatives = new ArrayList<Integer>(m_data.size() - classStats.nominalCounts[posLabel]);
//
//      for (int i = 0; i < m_data.size(); i++) {
//          if (m_data.get(i).classValue() == posLabel) {
//              positives.add(i);
//          } else {
//              negatives.add(i);
//          }
//      }
//
      numPosIn = dm.posCount();
      numNegIn = dm.negCount();


      buildClassifiers();

      // calc OOB error?
      if (getCalcOutOfBag()) {
          double outOfBagCount = 0.0;
          double errorSum = 0.0;
          boolean numeric = m_data.classAttribute().isNumeric();

          for (int i = 0; i < m_data.numInstances(); i++) {
              double vote;
              double[] votes;
              if (numeric) {
                  votes = new double[1];
              } else {
                  votes = new double[m_data.numClasses()];
              }

              // determine predictions for instance
              int voteCount = 0;
              for (int j = 0; j < m_Classifiers.length; j++) {
                  if (m_inBag[j][i]) {
                      continue;
                  }

                  voteCount++;
//          double pred = m_Classifiers[j].classifyInstance(m_data.instance(i));
                  if (numeric) {
                      // votes[0] += pred;
                      votes[0] += m_Classifiers[j].classifyInstance(m_data.instance(i));
                  } else {
                      //  votes[(int) pred]++;
                      double[] newProbs = m_Classifiers[j].distributionForInstance(m_data.instance(i));
                      // average the probability estimates
                      for (int k = 0; k < newProbs.length; k++) {
                          votes[k] += newProbs[k];
                      }
                  }
              }

              // "vote"
              if (numeric) {
                  vote = votes[0];
                  if (voteCount > 0) {
                      vote /= voteCount;    // average
                  }
              } else {
                  if (Utils.eq(Utils.sum(votes), 0)) {
                  } else {
                      Utils.normalize(votes);
                  }
                  vote = Utils.maxIndex(votes);   // predicted class
              }

              // error for instance
              outOfBagCount += m_data.instance(i).weight();
              if (numeric) {
                  errorSum += StrictMath.abs(vote - m_data.instance(i).classValue())
                          * m_data.instance(i).weight();
              } else {
                  if (vote != m_data.instance(i).classValue()) {
                      errorSum += m_data.instance(i).weight();
                  }
              }
          }

          m_OutOfBagError = errorSum / outOfBagCount;
      } else {
          m_OutOfBagError = 0;
      }

      // save memory
      m_data = null;
    }

    /**
     * Calculates the class membership probabilities for the given test
     * instance.
     *
     * @param instance the instance to be classified
     * @return preedicted class probability distribution
     * @throws Exception if distribution can't be computed successfully
     */
    public double[] distributionForInstance(Instance instance) throws Exception {

        double[] sums = new double[instance.numClasses()], newProbs;

        for (int i = 0; i < m_NumIterations; i++) {
            if (instance.classAttribute().isNumeric() == true) {
                sums[0] += m_Classifiers[i].classifyInstance(instance);
            } else {
                newProbs = m_Classifiers[i].distributionForInstance(instance);
                for (int j = 0; j < newProbs.length; j++) {
                    sums[j] += newProbs[j];
                }
            }
        }
        if (instance.classAttribute().isNumeric() == true) {
            sums[0] /= (double) m_NumIterations;
            return sums;
        } else if (Utils.eq(Utils.sum(sums), 0)) {
            return sums;
        } else {
            Utils.normalize(sums);
            return sums;
        }
    }

    /**
     * Returns description of the bagged classifier.
     *
     * @return description of the bagged classifier as a string
     */
    public String toString() {

        if (m_Classifiers == null) {
            return "Bagging: No model built yet.";
        }
        StringBuffer text = new StringBuffer();
        text.append("All the base classifiers: \n\n");
        for (int i = 0; i < m_Classifiers.length; i++) {
            text.append(m_Classifiers[i].toString() + "\n\n");
        }

        if (m_CalcOutOfBag) {
            text.append("Out of bag error: "
                    + Utils.doubleToString(m_OutOfBagError, 4)
                    + "\n\n");
        }

        return text.toString();
    }

    /**
     * Returns the revision string.
     *
     * @return	the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8034 $");
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

  /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
//    public static void main(String[] argv) {
//    }
}
