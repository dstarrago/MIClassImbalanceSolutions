/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package dataPreprocessing;
// Este es originalmente el MISMOTE_v01s1 del proyecto MILPro2

import java.util.*;
import java.util.ArrayList;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.filters.unsupervised.attribute.AddValues;


/**
 <!-- globalinfo-start -->
 * Resamples a dataset by applying the Synthetic Minority Oversampling TEchnique (SMOTE). The original dataset must fit entirely in memory. The amount of SMOTE and number of nearest neighbors may be specified. For more information, see <br/>
 * <br/>
 * Nitesh V. Chawla et. al. (2002). Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research. 16:321-357.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{al.2002,
 *    author = {Nitesh V. Chawla et. al.},
 *    journal = {Journal of Artificial Intelligence Research},
 *    pages = {321-357},
 *    title = {Synthetic Minority Over-sampling Technique},
 *    volume = {16},
 *    year = {2002}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -S &lt;num&gt;
 *  Specifies the random number seed
 *  (default 1)</pre>
 *
 * <pre> -P &lt;percentage&gt;
 *  Specifies percentage of SMOTE instances to create.
 *  (default 100.0)
 * </pre>
 *
 * <pre> -K &lt;nearest-neighbors&gt;
 *  Specifies the number of nearest neighbors to use.
 *  (default 5)
 * </pre>
 *
 * <pre> -C &lt;DD-selIndex&gt;
 *  Specifies the selIndex of the nominal class DD to SMOTE
 *  (default 0: auto-detect non-empty minority class))
 * </pre>
 *
 <!-- options-end -->
 *
 *
 * @author Danel
 * @version $Revision: 0001 $
 */
public class MWMOTE4MIL
  extends Filter
  implements SupervisedFilter, OptionHandler, TechnicalInformationHandler, MultiInstanceCapabilitiesHandler {

  /** for serialization. */
  static final long serialVersionUID = -1653880819059250364L;

  /** the random seed to use. */
  protected int m_RandomSeed = 1;

  /** the percentage of SMOTE instances to create. By default, makes IR == 1 */
  protected double m_Percentage = 0;
  
  /** number of examples to create*/
  protected int numSynthetics;

  /** the selIndex of the class DD. */
  protected String m_ClassValueIndex = "0";

  private MwmoteDataManager dm;

  /** whether to detect the minority class automatically. */
  protected boolean m_DetectMinorityClass = true;

  /** Indices of string attributes in the bag */
  protected StringLocator m_BagStringAtts = null;

  /** Indices of relational attributes in the bag */
  protected RelationalLocator m_BagRelAtts = null;

  public int numSynthetics() {
    if (numSynthetics <= 0) {
      return dm.negCount() - dm.posCount();
    }
    return numSynthetics;
  }

  public void setNumSynthetics(int n_Value) {
    this.numSynthetics = n_Value;
  }

  public void setDataManager(MwmoteDataManager dm) {
    this.dm = dm;
  }

  /**
   * Returns a string describing this classifier.
   *
   * @return 		a description of the classifier suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Resamples a dataset by applying the Synthetic Minority Oversampling TEchnique (SMOTE)." +
    " The original dataset must fit entirely in memory." +
    " The amount of SMOTE and number of nearest neighbors may be specified." +
    " For more information, see \n\n"
    + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   *
   * @return 		the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);

    result.setValue(Field.AUTHOR, "Nitesh V. Chawla et. al.");
    result.setValue(Field.TITLE, "Synthetic Minority Over-sampling Technique");
    result.setValue(Field.JOURNAL, "Journal of Artificial Intelligence Research");
    result.setValue(Field.YEAR, "2002");
    result.setValue(Field.VOLUME, "16");
    result.setValue(Field.PAGES, "321-357");

    return result;
  }

  /**
   * Returns the revision string.
   *
   * @return 		the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 0001 $");
  }

  /**
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // other
    result.enable(Capability.ONLY_MULTIINSTANCE);

    return result;
  }

  /**
   * Returns the capabilities of this multi-instance filter for the
   * relational data (i.e., the bags).
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getMultiInstanceCapabilities() {
    Capabilities result = new Capabilities(this);

    // attributes
    result.enableAllAttributes();
    result.disable(Capability.RELATIONAL_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enableAllClasses();
    result.enable(Capability.MISSING_CLASS_VALUES);
    result.enable(Capability.NO_CLASS);

    // other
    result.setMinimumNumberInstances(0);

    return result;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector newVector = new Vector();

    newVector.addElement(new Option(
	"\tSpecifies the random number seed\n"
	+ "\t(default 1)",
	"S", 1, "-S <num>"));

    newVector.addElement(new Option(
	"\tSpecifies percentage of SMOTE instances to create.\n"
	+ "\t(default 100.0)\n",
	"P", 1, "-P <percentage>"));

    newVector.addElement(new Option(
	"\tSpecifies the number of nearest neighbors to use.\n"
	+ "\t(default 5)\n",
	"K", 1, "-K <nearest-neighbors>"));

    newVector.addElement(new Option(
	"\tSpecifies the index of the nominal class value to SMOTE\n"
	+"\t(default 0: auto-detect non-empty minority class))\n",
	"C", 1, "-C <value-index>"));

    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   *
   <!-- options-start -->
   * Valid options are: <p/>
   *
   * <pre> -S &lt;num&gt;
   *  Specifies the random number seed
   *  (default 1)</pre>
   *
   * <pre> -P &lt;percentage&gt;
   *  Specifies percentage of SMOTE instances to create.
   *  (default 100.0)
   * </pre>
   *
   * <pre> -K &lt;nearest-neighbors&gt;
   *  Specifies the number of nearest neighbors to use.
   *  (default 5)
   * </pre>
   *
   * <pre> -C &lt;DD-selIndex&gt;
   *  Specifies the selIndex of the nominal class DD to SMOTE
   *  (default 0: auto-detect non-empty minority class))
   * </pre>
   *
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    String seedStr = Utils.getOption('S', options);
    if (seedStr.length() != 0) {
      setRandomSeed(Integer.parseInt(seedStr));
    } else {
      setRandomSeed(1);
    }

    String percentageStr = Utils.getOption('P', options);
    if (percentageStr.length() != 0) {
      setPercentage(new Double(percentageStr).doubleValue());
    } else {
      setPercentage(100.0);
    }

    String classValueIndexStr = Utils.getOption( 'C', options);
    if (classValueIndexStr.length() != 0) {
      setClassValue(classValueIndexStr);
    } else {
      m_DetectMinorityClass = true;
    }
  }

  /**
   * Gets the current settings of the filter.
   *
   * @return an array 	of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    Vector<String>	result;

    result = new Vector<String>();

    result.add("-C");
    result.add(getClassValue());

    result.add("-P");
    result.add("" + getPercentage());

    result.add("-S");
    result.add("" + getRandomSeed());

    return result.toArray(new String[result.size()]);
  }

  /**
   * Returns the tip text for this property.
   *
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String randomSeedTipText() {
    return "The seed used for random sampling.";
  }

  /**
   * Gets the random number seed.
   *
   * @return 		the random number seed.
   */
  public int getRandomSeed() {
    return m_RandomSeed;
  }

  /**
   * Sets the random number seed.
   *
   * @param DD 	the new random number seed.
   */
  public void setRandomSeed(int value) {
    m_RandomSeed = value;
  }

  /**
   * Returns the tip text for this property.
   *
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String percentageTipText() {
    return "The percentage of SMOTE instances to create.";
  }

  /**
   * Sets the percentage of SMOTE instances to create.
   *
   * @param DD	the percentage to use
   */
  public void setPercentage(double value) {
    if (value >= 0)
      m_Percentage = value;
    else
      System.err.println("Percentage must be >= 0!");
  }

  /**
   * Gets the percentage of SMOTE instances to create.
   *
   * @return 		the percentage of SMOTE instances to create
   */
  public double getPercentage() {
    return m_Percentage;
  }

  /**
   * Returns the tip text for this property.
   *
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String nearestNeighborsTipText() {
    return "The number of nearest neighbors to use.";
  }

  /**
   * Returns the tip text for this property.
   *
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String classValueTipText() {
    return "The index of the class value to which SMOTE should be applied. " +
    "Use a value of 0 to auto-detect the non-empty minority class.";
  }

  /**
   * Sets the selIndex of the class DD to which SMOTE should be applied.
   *
   * @param DD	the class DD selIndex
   */
  public void setClassValue(String value) {
    m_ClassValueIndex = value;
    if (m_ClassValueIndex.equals("0")) {
      m_DetectMinorityClass = true;
    } else {
      m_DetectMinorityClass = false;
    }
  }

  /**
   * Gets the selIndex of the class DD to which SMOTE should be applied.
   *
   * @return 		the selIndex of the clas DD to which SMOTE should be applied
   */
  public String getClassValue() {
    return m_ClassValueIndex;
  }

  /**
   * Sets the format of the input instances.
   *
   * @param instanceInfo 	an Instances object containing the input
   * 				instance structure (any instances contained in
   * 				the object are ignored - only the structure is required).
   * @return 			true if the outputFormat may be collected immediately
   * @throws Exception 		if the input format can't be set successfully
   */
  @Override
  public boolean setInputFormat(Instances instanceInfo) throws Exception {
    super.setInputFormat(instanceInfo);
    super.setOutputFormat(instanceInfo);
    m_BagStringAtts = new StringLocator(instanceInfo.attribute(1).relation());
    m_BagRelAtts    = new RelationalLocator(instanceInfo.attribute(1).relation());
    return true;
  }

  /**
   * Input an instance for filtering. Filter requires all
   * training instances be read before producing output.
   *
   * @param instance 		the input instance
   * @return 			true if the filtered instance may now be
   * 				collected with output().
   * @throws IllegalStateException if no input structure has been defined
   */
  @Override
  public boolean input(Instance instance) {
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      m_NewBatch = false;
    }
    if (m_FirstBatchDone) {
      push(instance);
      return true;
    } else {
      bufferInput(instance);
      return false;
    }
  }

  /**
   * Signify that this batch of input to the filter is finished.
   * If the filter requires all instances prior to filtering,
   * output() may now be called to retrieve the filtered instances.
   *
   * @return 		true if there are instances pending output
   * @throws IllegalStateException if no input structure has been defined
   * @throws Exception 	if provided options cannot be executed
   * 			on input instances
   */
  @Override
  public boolean batchFinished() throws Exception {
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }

    if (!m_FirstBatchDone) {
      // Do SMOTE, and clear the input instances.
      doSMOTE();
    }
    flushInput();

    m_NewBatch = true;
    m_FirstBatchDone = true;
    return (numPendingOutput() != 0);
  }

  /**
   * Determine which class MISMOTE should be applied
   * @return class selIndex where MISMOTE should be applied
   */
  private int classToSmote() throws Exception {
    Instances input = getInputFormat();
    int minIndex = 0;               // selIndex for the class where MISMOTE will be applied to
    if (m_DetectMinorityClass) {
      // find minority class
      AttributeStats classStats = input.attributeStats(input.classIndex());
      minIndex = Utils.minIndex(classStats.nominalCounts);
    } else {
      String classVal = getClassValue();
      if (classVal.equalsIgnoreCase("first")) {
        minIndex = 1;
      } else if (classVal.equalsIgnoreCase("last")) {
        minIndex = input.numClasses();
      } else {
        minIndex = Integer.parseInt(classVal);
      }
      if (minIndex > input.numClasses()) {
        throw new Exception("value index must be <= the number of classes");
      }
      minIndex--; // make it an selIndex
    }
    return minIndex;
  }

private int minClassSize() {
  Instances input = getInputFormat();
  int min = Integer.MAX_VALUE;
  int[] classCounts = input.attributeStats(input.classIndex()).nominalCounts;
  for (int i = 0; i < classCounts.length; i++) {
    if (classCounts[i] != 0 && classCounts[i] < min) {
      min = classCounts[i];
    }
  }
  return min;
}

/**
 * Output all dataset instances.
 */
private void pushInput() {
  //Instances output = getOutputFormat();
  Instances input = getInputFormat();
  Enumeration instanceEnum = input.enumerateInstances();
  while(instanceEnum.hasMoreElements()) {
    Instance instance = (Instance) instanceEnum.nextElement();
    //The following weighting options should be implemented soon
    // Options: each bag has 
    // a) weight = 1
    // bagProd) weight = sizeNewBag
    // For now, set the weight of the bag to 1
    instance.setWeight(1);
    // copy string values etc. from input to output
    //copyValues(instance, false, instance.dataset(), getOutputFormat());
    push(instance);
    //copyValues(instance, false);
    //output.add(instance);
  }
}

private void arrangeOutputFormat(int numNewBags) {
  StringBuilder labels = new StringBuilder();
  for (int i = 0; i < numNewBags; i++) {
    labels.append("sm").append(String.valueOf((int) i));
    if (i < numNewBags - 1)
      labels.append(",");
  }
  String str = labels.toString();
  AddValues av = new AddValues();
  Instances input = getInputFormat();
  try {
    av.setAttributeIndex("first");
    av.setSort(false);
    av.setLabels(str);
    av.setInputFormat(input);
    Instances output = Filter.useFilter(input, av);
    super.setOutputFormat(output);
  } catch (Exception e) {
    System.out.println("Problems adding labels to BagID attribute : " + e.getMessage());
  }
}

  /**
   * Creates a new dataset of the same size using random sampling
   * with replacement according to the given weight vector. The
   * weights of the instances in the new dataset are set to one.
   * The length of the weight vector has to be the same as the
   * number of instances in the dataset, and all weights have to
   * be positive.
   *
   * @param random a random number generator
   * @param weights the weight vector
   * @return the new dataset
   * @throws IllegalArgumentException if the weights array is of the wrong
   * length or contains negative weights.
   */
  public ArrayList<Integer> resampleWithWeights(Random random, 
          ArrayList<Integer> posIndex, double[] weights) {

    if (weights.length != posIndex.size()) {
      throw new IllegalArgumentException("weights.length != posIndex.size()");
    }
    ArrayList<Integer> sortedPos = new ArrayList<Integer>(posIndex.size());
    if (posIndex.isEmpty()) {
      return sortedPos;
    }
    double[] probabilities = new double[posIndex.size()];
    double sumProbs = 0, sumOfWeights = Utils.sum(weights);
    for (int i = 0; i < posIndex.size(); i++) {
      sumProbs += random.nextDouble();
      probabilities[i] = sumProbs;
    }
    Utils.normalize(probabilities, sumProbs / sumOfWeights);

    // Make sure that rounding errors don't mess things up
    probabilities[posIndex.size() - 1] = sumOfWeights;
    int k = 0; int l = 0;
    sumProbs = 0;
    while ((k < posIndex.size() && (l < posIndex.size()))) {
      if (weights[l] < 0) {
        throw new IllegalArgumentException("Weights have to be positive.");
      }
      sumProbs += weights[l];
      while ((k < posIndex.size()) && (probabilities[k] <= sumProbs)) {
        sortedPos.add(posIndex.get(l));
        k++;
      }
      l++;
    }
    return sortedPos;
  }

//  public ArrayList<Integer> posBagDistributionXP(Random random, int posIndex) {
//    BigDecimal[] weights = dm.diverseDensity(posIndex);
//    ArrayList<Integer> sortedPos = new ArrayList<Integer>(weights.length);
//    BigDecimal sumOfWeights = new BigDecimal(0, MathContext.DECIMAL32); // Utils.sum(weights);
//    for (int i = 0; i < weights.length; i++) {
//      sumOfWeights = sumOfWeights.add(weights[i], MathContext.DECIMAL32);    // Efficiency issue? (SUM)
//    }
//    if (sumOfWeights.compareTo(new BigDecimal(0, MathContext.DECIMAL32)) == 0) {
//      for (int i = 0; i < weights.length; i++) {
//        sortedPos.add(i);
//      }
//      //System.out.println("sumOfWeights from DD is zero!");
//      return sortedPos;
//    }
//
//    BigDecimal[] probabilities = new BigDecimal[weights.length];
//    double sumProbs = 0;
//    for (int i = 0; i < weights.length; i++) {
//      sumProbs += random.nextDouble();
//      probabilities[i] = new BigDecimal(sumProbs, MathContext.DECIMAL32);    // Efficiency issue?    (CONVERSION)
//    }
//    // Normalize probabilities
//    BigDecimal ratio = sumOfWeights.divide(new BigDecimal(sumProbs, MathContext.DECIMAL32), MathContext.DECIMAL32);
////    Utils.normalize(probabilities, sumProbs / sumOfWeights);
//    for (int i = 0; i < probabilities.length; i++) {
//      probabilities[i] = probabilities[i].multiply(ratio, MathContext.DECIMAL32);   // Efficiency issue?  (DIVISION)
//    }
//    // Make sure that rounding errors don't mess things up
//    probabilities[weights.length - 1] = sumOfWeights;
//    int k = 0; int l = 0;
//    BigDecimal sumPr = new BigDecimal(0, MathContext.DECIMAL32);
//    while ((k < weights.length && (l < weights.length))) {
////      if (weights[l].compareTo(new BigDecimal(0)) < 0) {      // Efficiency issue? (COMPARATION)
////        throw new IllegalArgumentException("Weights have to be positive.");
////      }
//      sumPr = sumPr.add(weights[l], MathContext.DECIMAL32);
//      while ((k < weights.length) && (probabilities[k].compareTo(sumPr) <= 0)) {    // Efficiency issue?  (COMPARATION)
//        sortedPos.add(l);
//        k++;
//      }
//      l++;
//    }
//    return sortedPos;
//  }

  public ArrayList<Integer> posBagDistribution(Random random, int posIndex) {
    double[] weights = dm.diverseDensity(posIndex);
    ArrayList<Integer> sortedPos = new ArrayList<Integer>(weights.length);
    double sumOfWeights = Utils.sum(weights);
    if (sumOfWeights == 0) {
      for (int i = 0; i < weights.length; i++) {
        sortedPos.add(i);
      }
      //System.out.println("sumOfWeights from DD is zero!");
      return sortedPos;
    }

    double[] probabilities = new double[weights.length];
    double sumProbs = 0;
    for (int i = 0; i < weights.length; i++) {
      sumProbs += random.nextDouble();
      probabilities[i] = sumProbs;    // Efficiency issue?    (CONVERSION)
    }
    // Normalize probabilities
    Utils.normalize(probabilities, sumProbs / sumOfWeights);
    // Make sure that rounding errors don't mess things up
    probabilities[weights.length - 1] = sumOfWeights;
    int k = 0; int l = 0;
    sumProbs = 0;
    while ((k < weights.length && (l < weights.length))) {
      if (weights[l] < 0) {      // Efficiency issue? (COMPARATION)
        throw new IllegalArgumentException("Weights have to be positive.");
      }
      sumProbs += weights[l];
      while ((k < weights.length) && (probabilities[k] <= sumProbs)) {    // Efficiency issue?  (COMPARATION)
        sortedPos.add(l);
        k++;
      }
      l++;
    }
    return sortedPos;
  }

  /**
   * The procedure implementing the SMOTE algorithm. The output
   * instances are pushed onto the output queue for collection.
   *
   * @throws Exception 	if provided options cannot be executed
   * 			on input instances
   */
  protected void doSMOTE() throws Exception {

    Instances input = getInputFormat();

    if (dm == null) {
      dm = new MwmoteDataManager(input);
    }

    /**
     * Header info for the bag
     */
    Instances bagInsts = input.attribute(1).relation();

    /**
     * Determine the amount of MISMOTE to apply
     */    
    int numNewBags = numSynthetics();                      // Total number of bags to generate

    //**************************************************************************

//    // Algorithm parameters
//    int K1 = 5;  // 5
//    int K2 = 3;  // 3
//    int K3 = positives.size() / 2;
    /**
     * Number of nearest neighbors to be used
     */
//    int nearestNeighbors = numNearestNeighbors(minClassSize);

    //**************************************************************************

    /**
     * Index of the new synthetic bag
     */
    double newBagIndex = input.attribute(0).numValues();

    /**
     * Arrange and set the output format
     */
    arrangeOutputFormat(numNewBags);

    /**
     * Push the input instances to the output
     */
    pushInput(); 
    
    /*
    // compute Value Distance Metric matrices for nominal features
    Map vdmMap = new HashMap();
    Enumeration attrEnum = getInputFormat().enumerateAttributes();
    while(attrEnum.hasMoreElements()) {
      Attribute attr = (Attribute) attrEnum.nextElement();
      if (!attr.equals(getInputFormat().classAttribute())) {
	if (attr.isNominal() || attr.isString()) {
	  double[][] vdm = new double[attr.numValues()][attr.numValues()];
	  vdmMap.put(attr, vdm);
	  int[] featureValueCounts = new int[attr.numValues()];
	  int[][] featureValueCountsByClass = new int[getInputFormat().classAttribute().numValues()][attr.numValues()];
	  instanceEnum = getInputFormat().enumerateInstances();
	  while(instanceEnum.hasMoreElements()) {
	    Instance instance = (Instance) instanceEnum.nextElement();
	    int DD = (int) instance.DD(attr);
	    int classValue = (int) instance.classValue();
	    featureValueCounts[DD]++;
	    featureValueCountsByClass[classValue][DD]++;
	  }
	  for (int valueIndex1 = 0; valueIndex1 < attr.numValues(); valueIndex1++) {
	    for (int valueIndex2 = 0; valueIndex2 < attr.numValues(); valueIndex2++) {
	      double sum = 0;
	      for (int classValueIndex = 0; classValueIndex < getInputFormat().numClasses(); classValueIndex++) {
		double c1i = (double) featureValueCountsByClass[classValueIndex][valueIndex1];
		double c2i = (double) featureValueCountsByClass[classValueIndex][valueIndex2];
		double c1 = (double) featureValueCounts[valueIndex1];
		double c2 = (double) featureValueCounts[valueIndex2];
		double term1 = c1i / c1;
		double term2 = c2i / c2;
		sum += Math.abs(term1 - term2);
	      }
	      vdm[valueIndex1][valueIndex2] = sum;
	    }
	  }
	}
      }
    }
    */
    // use this random source for all required randomness
    Random random = new Random(getRandomSeed());
    
    /**
     * The main loop to handle computing nearest neighbors and generating
     * SMOTE examples from each bag in the original minority class data
     */
    for (int i = 0; i < numNewBags; i++) {

      ArrayList<Integer> distribution = resampleWithWeights(random, dm.inforMin(), dm.selectionProb());
      // Select a random example from the list of informative positive examples
      int selIndex = random.nextInt(distribution.size()); // OJO: cambiar distribución uniforme por SelectionProb
      // index of the selected example in the positive example index list
      int indexBag1 = distribution.get(selIndex);
      Instance bag1 = dm.posBag(indexBag1);

      ArrayList<Integer> cluster = dm.clusterWithPosBag(indexBag1);
      selIndex = random.nextInt(cluster.size());
      int indexBag2 = cluster.get(selIndex);
      Instance bag2 = dm.posBag(indexBag2);

      // Determine the sizeNewBag of the synthetic bag as the average of the two bag sizes
      int sizeBag1 = bag1.relationalValue(1).numInstances();
      int sizeBag2 = bag2.relationalValue(1).numInstances();
      int sizeNewBag = (sizeBag1 + sizeBag2) / 2;

      /*
       * Create a new bag from bag1 and bag2.
       * The new bag is an instance with an ID attribute, a relational attribute
       * and a class label.
       * Inside the new bag create N instances.
       */
      int s = sizeNewBag;
      //System.out.println();
      //System.out.println("*******   New Bag " + (int) newBagIndex);
      //System.out.println();
      while (s > 0) {
        /**
         * Create an instance of the new bag
         */

        /**
         * Choose a random number between 0 (inclusive) and sizeBag1 (exclusive),
         * call it indexInst1.
         */
        ArrayList<Integer> distBag1 = posBagDistribution(random, indexBag1);
        selIndex = random.nextInt(sizeBag1);
        int indexInst1 = distBag1.get(selIndex);
//        int indexInst1 = 1;

        /**
         * Choose a random number between 0 (inclusive) and sizeBag2 (exclusive),
         * call it indexInst2.
         */
        ArrayList<Integer> distBag2 = posBagDistribution(random, indexBag2);
        selIndex = random.nextInt(sizeBag2);
        int indexInst2 = distBag2.get(selIndex);
//        int indexInst2 = 1;

        double[] values = new double[bagInsts.numAttributes()];
        Enumeration attrEnum = bagInsts.enumerateAttributes();
        while(attrEnum.hasMoreElements()) {
          Attribute attr = (Attribute) attrEnum.nextElement();
          if (attr.isNumeric()) {
            double nnVal = bag2.relationalValue(1).instance(indexInst2).value(attr);
            double iVal = bag1.relationalValue(1).instance(indexInst1).value(attr);
            double dif = nnVal - iVal;
            double gap = random.nextDouble();
            values[attr.index()] = (double) (iVal + gap * dif);
          } else if (attr.isDate()) {
            double nnVal = bag2.relationalValue(1).instance(indexInst2).value(attr);
            double iVal = bag1.relationalValue(1).instance(indexInst1).value(attr);
            double dif = nnVal - iVal;
            double gap = random.nextDouble();
            values[attr.index()] = (long) (iVal + gap * dif);
          } else {
            /**
             * In the case where the attribute is nominal we take one instance
             * at random from each bag in the K nearest neighbors (also from
             * the actual bag bag1) and compute the attribute DD more
             * frequently seen.
             */
            int[] valueCounts = new int[attr.numValues()];
            int iVal = (int) bag1.relationalValue(1).instance(indexInst1).value(attr);
            valueCounts[iVal]++;
            for (int nnEx = 0; nnEx < dm.numNNAttrVoting(); nnEx++) {
//              Instance nnBag = input.get(positives.get(kNN[inforMin.get(selIndex)].get(nnEx)));
              Instance nnBag = dm.posBag(dm.NNToPos(dm.inforMin(selIndex)).get(nnEx));
              int nnNumInst = nnBag.relationalValue(1).numInstances();
              int nnInst = random.nextInt(nnNumInst);
              int val = (int) nnBag.relationalValue(1).instance(nnInst).value(attr);
              valueCounts[val]++;
            }
            int maxIndex = 0;
            int max = Integer.MIN_VALUE;
            for (int index = 0; index < attr.numValues(); index++) {
              if (valueCounts[index] > max) {
              max = valueCounts[index];
              maxIndex = index;
              }
            }
            values[attr.index()] = maxIndex;
          }
        }
        Instance syntheticInst = new DenseInstance(1.0, values);
        //System.out.println(syntheticInst.toString());
        bagInsts.add(syntheticInst);
        s--;
      }
      double bagWeight = 1;
      addBag( bagInsts, (int) newBagIndex, dm.posClassLabel(), bagWeight);
      bagInsts   = bagInsts.stringFreeStructure();
      newBagIndex++;
    }
//    System.out.println();
//    System.out.println("Diverse Density Matrix:");
//    System.out.println(Utils.arrayToString(diverseDensity));
//    System.out.println();
  }
  
  /**
   * adds a new bag out of the given data and adds it to the output
   *
   * @param input       the intput dataset
   * @param output      the dataset this bag is added to
   * @param bagInsts    the instances in this bag
   * @param bagIndex    the bagIndex of this bag
   * @param classValue  the associated class DD
   * @param bagWeight   the weight of the bag
   */
  protected void addBag(
      Instances bagInsts,
      int bagIndex,
      double classValue,
      double bagWeight) {

    Instances output = outputFormatPeek();
    int value = output.attribute(1).addRelation(bagInsts);
    Instance newBag = new DenseInstance(output.numAttributes());
    newBag.setValue(0, bagIndex);
    newBag.setValue(2, classValue);
    newBag.setValue(1, value);
    newBag.setWeight(bagWeight);
    newBag.setDataset(output);
    output.add(newBag);
    push(newBag);
  }

  /**
   * Main method for running this filter.
   *
   * @param args 	should contain arguments to the filter:
   * 			use -h for help
   */
  public static void main(String[] args) {
    runFilter(new MWMOTE4MIL(), args);
  }
}
