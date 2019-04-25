/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package dataPreprocessing;

import Utils.CosineDist;
import Utils.InstanceLevelDistance;
import java.util.*;
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
 * <pre> -C &lt;value-index&gt;
 *  Specifies the index of the nominal class value to SMOTE
 *  (default 0: auto-detect non-empty minority class))
 * </pre>
 *
 <!-- options-end -->
 *
 *
 * @author Danel
 * @version $Revision: 0001 $
 */
public class BagSMOTE
  extends Filter
  implements SupervisedFilter, OptionHandler, TechnicalInformationHandler, MultiInstanceCapabilitiesHandler {

  /** for serialization. */
  static final long serialVersionUID = -1653880819059250364L;

  /** the number of neighbors to use. */
  protected int m_NearestNeighbors = 5;

  /** the random seed to use. */
  protected int m_RandomSeed = 1;

  /** the percentage of SMOTE instances to create. By default, makes IR == 1 */
  protected double m_Percentage = 0;
  
  /**the number of SMOTE instances to create*/
  protected int numSynthetics=0;

  /** the index of the class value. */
  protected String m_ClassValueIndex = "0";

  /** whether to detect the minority class automatically. */
  protected boolean m_DetectMinorityClass = true;

  /** Indices of string attributes in the bag */
  protected StringLocator m_BagStringAtts = null;

  /** Indices of relational attributes in the bag */
  protected RelationalLocator m_BagRelAtts = null;

  private InstanceLevelDistance cosineDist = new CosineDist();

  /**
   * List of positive example indexes in the training set.
   */
  private ArrayList<Integer> positives;

  /**
   * List of negative example indexes in the training set.
   */
  private ArrayList<Integer> negatives;

  /**
   * Index of the positive class label
   */
  private int posClassLabel;

  private AttributeStats classStats;

  private Instance[] posInstPool;

    public int numSynthetics() {
      if (numSynthetics <= 0) {
        return negatives.size() - positives.size();
      }
      return numSynthetics;
    }

    public void setNumSynthetics(int n_Value) {
        this.numSynthetics = n_Value;
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
   * relational data (indexBag1.e., the bags).
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
   * <pre> -C &lt;value-index&gt;
   *  Specifies the index of the nominal class value to SMOTE
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

    String nnStr = Utils.getOption('K', options);
    if (nnStr.length() != 0) {
      setNearestNeighbors(Integer.parseInt(nnStr));
    } else {
      setNearestNeighbors(5);
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

    result.add("-K");
    result.add("" + getNearestNeighbors());

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
   * @param value 	the new random number seed.
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
   * @param value	the percentage to use
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
   * Sets the number of nearest neighbors to use.
   *
   * @param value	the number of nearest neighbors to use
   */
  public void setNearestNeighbors(int value) {
    if (value >= 1)
      m_NearestNeighbors = value;
    else
      System.err.println("At least 1 neighbor necessary!");
  }

  /**
   * Gets the number of nearest neighbors to use.
   *
   * @return 		the number of nearest neighbors to use
   */
  public int getNearestNeighbors() {
    return m_NearestNeighbors;
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
   * Sets the index of the class value to which SMOTE should be applied.
   *
   * @param value	the class value index
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
   * Gets the index of the class value to which SMOTE should be applied.
   *
   * @return 		the index of the clas value to which SMOTE should be applied
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
   * @return class index where MISMOTE should be applied
   */
  private int classToSmote() throws Exception {
    Instances input = getInputFormat();
    int minIndex = 0;               // index for the class where MISMOTE will be applied to
    if (m_DetectMinorityClass) {
      // find minority class
      if (classStats == null) {
        classStats = input.attributeStats(input.classIndex());
      }
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
      minIndex--; // make it an index
    }
    return minIndex;
  }

  /**
   * Return the number of nearest neighbors to be used. This number will be less
   * than the number of examples in the minority class. An exception is raised 
   * if the number of examples in the minority class is only 1.
   * 
   * @param totalInstanceNumber Size of the minority class
   * @return number of nearest neighbors
   * @throws java.lang.Exception
   */
  private int numNearestNeighbors(int totalInstanceNumber) throws Exception {
    int min = (m_DetectMinorityClass)? totalInstanceNumber: Integer.MAX_VALUE;
    int nearestNeighbors;               // Number of nearest neighbors to be used
    if (min <= getNearestNeighbors()) {
      nearestNeighbors = min - 1;
    } else {
      nearestNeighbors = getNearestNeighbors();
    }
    if (nearestNeighbors < 1)
      throw new Exception("Cannot use 0 neighbors!");
    return nearestNeighbors;
  }

  /**
   * Find the set of extra indices to use if the percentage is not evenly
   * divisible by 100.
   *
   * @param numSamples number of examples in the minority class where MISMOTE
   * will be applied.
   */
private Set findExtraIndices(int numSamples, int numExtraSamples) {
  Random rand = new Random(getRandomSeed());

  List extraIndices = new LinkedList();
  if (numExtraSamples >= 1) {
    for (int i = 0; i < numSamples; i++) {
      extraIndices.add(i);
    }
  }
  Collections.shuffle(extraIndices, rand);
  extraIndices = extraIndices.subList(0, numExtraSamples);
  Set extraIndexSet = new HashSet(extraIndices);
  return extraIndexSet;
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
    // b) weight = sizeNewBag
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
   * The procedure implementing the SMOTE algorithm. The output
   * instances are pushed onto the output queue for collection.
   *
   * @throws Exception 	if provided options cannot be executed
   * 			on input instances
   */
  protected void doSMOTE() throws Exception {
    Instances input = getInputFormat();
    /**
     * Header info for the bag
     */
    Instances bagInsts = input.attribute(1).relation();

    classStats = input.attributeStats(input.classIndex());
    /**
     * Determine which class MISMOTE will be applied to
     */
    posClassLabel = classToSmote();
    /**
     * Size of the minority class
     */
    int minClassSize = classStats.nominalCounts[posClassLabel];
    
    positives = new ArrayList<Integer>(minClassSize);
    negatives = new ArrayList<Integer>(input.size() - minClassSize);
    int posInstPoolSize = 0;
    for (int i = 0; i < input.size(); i++) {
      Instance bag = input.get(i);
      if (bag.classValue() == posClassLabel) {
        positives.add(i);
        posInstPoolSize += bag.relationalValue(1).size();
      } else {
        negatives.add(i);
      }
    }

    /**
     * Determine the amount of MISMOTE to apply
     */
    int numNewBags = numSynthetics();                      // Total number of bags to generate
    int SMOTExSample = numNewBags / minClassSize;       // Number of bags to generate for each original bag
    int extraSamplesCount = numNewBags % minClassSize;  // Additional number of bags to generate from random original bags

    int instIndex = 0;
    posInstPool = new Instance[posInstPoolSize];
    for (int i = 0; i < positives.size(); i++) {
      Instance bag = input.get(positives.get(i));
      for (int j = 0; j < bag.relationalValue(1).size(); j++) {
        posInstPool[instIndex++] = bag.relationalValue(1).get(j);
      }
    }

    double[][] interInstancesDistances = new double[posInstPoolSize][posInstPoolSize];
    for (int i = 0; i < posInstPoolSize; i++) {
      Instance x = posInstPool[i];
      interInstancesDistances[i][i] = 1;
      for (int j = 0; j < i; j++) {
        Instance y = posInstPool[j];
        interInstancesDistances[i][j] = cosineDist.measure(x, y);   // Measures.cosDistance(x, y);
        interInstancesDistances[j][i] = interInstancesDistances[i][j];
      }
    }

    int[][] KNNindex = new int[posInstPoolSize][];
    for (int i = 0; i < posInstPoolSize; i++) {
      KNNindex[i] = Utils.sort(interInstancesDistances[i]);
    }

    /**
     * Number of nearest neighbors to be used
     */
    int nearestNeighbors = numNearestNeighbors(posInstPoolSize);

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
	    int value = (int) instance.value(attr);
	    int classValue = (int) instance.classValue();
	    featureValueCounts[value]++;
	    featureValueCountsByClass[classValue][value]++;
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

    // find the set of extra indices to use if the percentage is not evenly divisible by 100
    Set extraIndexSet = findExtraIndices(minClassSize, extraSamplesCount);

    int poolBagIndex = 0;
    /**
     * The main loop to handle computing nearest neighbors and generating
     * SMOTE examples from each bag in the original minority class data
     */
    for (int indexBag1 = 0; indexBag1 < positives.size(); indexBag1++) {

      Instance bag1 = input.instance(positives.get(indexBag1));
      int sizeBag1 = bag1.relationalValue(1).numInstances();

      int sizeNewBag = sizeBag1;

      // create synthetic examples
      int n = SMOTExSample;
      while(n > 0 || extraIndexSet.remove(indexBag1)) {

        for (int s = 0; s < sizeNewBag; s++) {
          Instance inst1 = bag1.relationalValue(1).get(s);
          int poolIndex = poolBagIndex + s;
          int selKNN = random.nextInt(nearestNeighbors);
          Instance inst2 = posInstPool[KNNindex[poolIndex][selKNN]];

          double[] values = new double[bagInsts.numAttributes()];
          Enumeration attrEnum = bagInsts.enumerateAttributes();
          while(attrEnum.hasMoreElements()) {
            Attribute attr = (Attribute) attrEnum.nextElement();
            if (attr.isNumeric()) {
              double val1 = inst1.value(attr);
              double val2 = inst2.value(attr);
              double dif = val2 - val1;
              double gap = random.nextDouble();
              values[attr.index()] = (double) (val1 + gap * dif);
            } else if (attr.isDate()) {
              double val1 = inst1.value(attr);
              double val2 = inst2.value(attr);
              double dif = val2 - val1;
              double gap = random.nextDouble();
              values[attr.index()] = (long) (val1 + gap * dif);
            } else {
              /**
               * In the case where the attribute is nominal we take one instance
               * at random from each bag in the K nearest neighbors (also from
               * the actual bag bag1) and compute the attribute value more
               * frequently seen.
               */
              int[] valueCounts = new int[attr.numValues()];
              int val1 = (int) inst1.value(attr);
              valueCounts[val1]++;
              for (int nnEx = 0; nnEx < nearestNeighbors; nnEx++) {
                Instance instNN = posInstPool[KNNindex[poolIndex][nnEx]];
                int valNN = (int) instNN.value(attr);
                valueCounts[valNN]++;
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
        }
        double bagWeight = 1;
        addBag( bagInsts, (int) newBagIndex, posClassLabel, bagWeight);
        bagInsts   = bagInsts.stringFreeStructure();
        newBagIndex++;
        n--;
      }
      poolBagIndex += sizeBag1;
    }
    //for (int indexBag1 = 0; indexBag1 < output.numInstances(); indexBag1++)
    //  push(output.instance(indexBag1));  }
  }
  
  /**
   * adds a new bag out of the given data and adds it to the output
   *
   * @param input       the intput dataset
   * @param output      the dataset this bag is added to
   * @param bagInsts    the instances in this bag
   * @param bagIndex    the bagIndex of this bag
   * @param classValue  the associated class value
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
    runFilter(new BagSMOTE(), args);
  }
}
