/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ensembles;

import dataPreprocessing.MISMOTE;
//import MISMOTE.MISMOTE;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.classifiers.mi.CitationKNN;
import weka.core.AdditionalMeasureProducer;
import weka.core.AttributeStats;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.RelationalLocator;
import weka.core.RevisionUtils;
import weka.core.StringLocator;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;

public class MISOBagging
        extends RandomizableParallelIteratedSingleClassifierEnhancer
        implements WeightedInstancesHandler, AdditionalMeasureProducer,
        TechnicalInformationHandler {

    private int sigma = 5;
    private int gamma = 50;
    private boolean negBoostrap = true;
    private boolean posBootstrap = true;
    private Instances negatives;
    private Instances positives;
    private int p;
    private int n;
    private int p1;
    private int m_RandomSeed = 1;

    //utilizando la clase MISMOTE del proyecto MILPro3
    private MISMOTE mismote;

    public int getM_RandomSeed() {
        return m_RandomSeed;
    }

    public void setM_RandomSeed(int m_RandomSeed) {
        this.m_RandomSeed = m_RandomSeed;
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
        return negBoostrap;
    }

    public void setNegBoostrap(boolean negBoostrap) {
        this.negBoostrap = negBoostrap;
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
    public MISOBagging() {

        m_Classifier = new CitationKNN();
        //mismote = new MISMOTE();
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
     * Options after -- are passed to the designated classifier.<p>
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

    protected Random m_random;
    protected boolean[][] m_inBag;
    protected Instances m_data;

    /**
     * Devuelve un cjto de instancias remuestreadas por bootstrap
     *
     * @param n numero aleatorio de instancias
     * @param D cjto de instancias a remuestrear
     * @return el cjto de instancias remuestreadas
     */
    private Instances bootstrap(int n, Instances D) {
        //se crea el cjto nuevo de instancias vacio
        Instances newData = new Instances(D, n);
        Random rand = new Random(m_Seed);
        for (int i = 0; i < n; i++) {
            //generar el num aleatorio entre 1 y la cant de instancias
            int r = rand.nextInt(D.numInstances());
            //añadir al nuevo cjto de instancias la instancia D[r]
            newData.add(D.get(r));
        }
        return newData;
    }


    /**
     * Inicializar todos los datos necesarios
     */
    private void inicializar() {
        // Determinar la clase positiva
        AttributeStats classStats = m_data.attributeStats(m_data.classIndex());
        int posLabel = Utils.minIndex(classStats.nominalCounts);

        negatives = new Instances(m_data, m_data.numInstances());
        positives = new Instances(m_data, classStats.nominalCounts[posLabel]);

        for (int i = 0; i < m_data.numInstances(); i++) {
            if (m_data.get(i).classValue() == posLabel) {
                positives.add(m_data.get(i));
                //System.out.println("POSITIVA: " + m_data.get(i).toString(m_data.get(i).classAttribute()));

            } else {
                negatives.add(m_data.get(i));
                //System.out.println("NEGATIVA: " + m_data.get(i).toString(m_data.get(i).classAttribute()));

            }
        }
        p = positives.numInstances();
        n = negatives.numInstances();
//        System.out.println("p = "+p);
//        System.out.println("n = "+n);

        if (sigma == 0) {
            p1 = n * gamma / (100 - gamma); //numero de ejemplos positivos de salida
        }

    }

    /**
     * @param iteration the number of the iteration for the requested training
     * set.
     * @return the training set for the supplied iteration number
     * @throws Exception if something goes wrong when generating a training set.
     */
    protected synchronized Instances getTrainingSet(int iteration) throws Exception {
        Instances bagData;
        //Inicializar los cjtos de clases positivas y negativas

        //Para cuando se submuestrea
        Instances DN;
        Instances DP;
        Instances ST;

        //Generar el num aleatorio como lo hacen en la clase Bagging
        Random rand = new Random(m_Seed + iteration);
        if (sigma > 0) {
            //lo de la dist normal
            double x = rand.nextGaussian();
//            System.out.println("x = "+ x);
            x = Math.abs(x);
            double Gamma = gamma + sigma * x;
            p1 = (int) (n * Gamma / (100 - Gamma));
        }

        System.out.println("P1: " + p1);
        DP = bootstrap(p, positives);

        mismote.setInputFormat(m_data);
        mismote.SetNumSynthetics(p1 - p);

        Instances syn = Filter.useFilter(m_data, mismote);
        ST = new Instances(syn, m_data.size(), syn.size() - m_data.size());

        DN = bootstrap(n, negatives);
//        System.out.println("DP.size()= " + DP.size());
//        System.out.println("ST.size()= " + ST.size());
//        System.out.println("DN.size()= " + DN.size());

        int size = p1 + p + n;
        //para hacer la union en el orden que viene descrita en el algoritmo
        bagData = new Instances(ST,size);
        bagData.addAll(DP);
        bagData.addAll(ST);
        bagData.addAll(DN);
        
//        int size=p1+p+n;
//        bagData=new Instances(ST,size);
//        for (int i=0;i<DP.size();i++)
//        {
//            Instance X=(Instance) DP.get(i).copy();
//            bagData.add(X);
//        }
//        for (int i=0;i<DN.size();i++)
//        {
//            Instance X=(Instance) DN.get(i).copy();
//            bagData.add(X);
//        }
//        
//        for (int i=m_data.size();i<ST.size();i++)
//        {
//            Instance X=ST.get(i);
//            bagData.add(X);
//        }

//        System.out.println("bagData.size() " + bagData.size());
        return bagData;

    }

    /**
     * Bagging method.
     *
     * @param data the training data to be used for generating the bagged
     * classifier.
     * @throws Exception if the classifier could not be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        m_data = new Instances(data);
        m_data.deleteWithMissingClass();
        inicializar();
        super.buildClassifier(m_data);

        if (m_CalcOutOfBag && (m_BagSizePercent != 100)) {
            throw new IllegalArgumentException("Bag size needs to be 100% if "
                    + "out-of-bag error is to be calculated!");
        }

        int bagSize = m_data.numInstances() * m_BagSizePercent / 100;
        m_random = new Random(m_Seed);

        m_inBag = null;
        if (m_CalcOutOfBag) {
            m_inBag = new boolean[m_Classifiers.length][];
        }

        for (int j = 0; j < m_Classifiers.length; j++) {
            if (m_Classifier instanceof Randomizable) {
                ((Randomizable) m_Classifiers[j]).setSeed(m_random.nextInt());
            }
        }

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

    private void addBag(Instances bagData, Instances header, int i, int i0, int i1) {
        Instances output = bagData;
        int value = output.attribute(1).addRelation(header);
        Instance newBag = new DenseInstance(output.numAttributes());
        newBag.setValue(0, i);
        newBag.setValue(2, i0 - 1);
        newBag.setValue(1, value);
        newBag.setWeight(i1);
        newBag.setDataset(output);
        output.add(newBag);
        push(newBag, bagData);
    }

    protected void push(Instance instance, Instances bagData) {

        if (instance != null) {
            if (instance.dataset() != null) {
                copyValues(instance, bagData);
            }
            instance.setDataset(bagData);
//      m_OutputQueue.push(instance);
        }
    }

    protected void copyValues(Instance instance, Instances bagData) {

        RelationalLocator.copyRelationalValues(
                instance,
                bagData,
                new RelationalLocator(bagData));

        StringLocator.copyStringValues(
                instance,
                bagData,
                new StringLocator(bagData));
    }

    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String[] argv) {
        runClassifier(new MISOBagging(), argv);
    }
}
