/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.MultiInstanceCapabilitiesHandler;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MultiInstanceToPropositional;

import java.util.Enumeration;
import java.util.Vector;

/**
 * A Wrapper method for applying standard propositional learners to multi-instance data
 * using the Standard MIL assumption.<br/>
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -A [0|1|2|3]
 *  The type of weight setting for each single-instance:
 *  0.keep the weight to be the same as the original value;
 *  1.weight = 1.0
 *  2.weight = 1.0/Total number of single-instance in the
 *   corresponding bag
 *  3. weight = Total number of single-instance / (Total
 *   number of bags * Total number of single-instance
 *   in the corresponding bag).
 *  (default: 1)</pre>
 *
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.rules.ZeroR)</pre>
 *
 * <pre>
 * Options specific to classifier weka.classifiers.rules.ZeroR:
 * </pre>
 *
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *
 <!-- options-end -->
 *
 * @author Danel
 */
public class MIStdWrapper 
  extends SingleClassifierEnhancer
  implements MultiInstanceCapabilitiesHandler, OptionHandler,
             TechnicalInformationHandler {

  private int positiveLabel = 1;

  /** for serialization */
  static final long serialVersionUID = -7707766152904315910L;

  /** Filter used to convert MI dataset into single-instance dataset */
  protected MultiInstanceToPropositional m_ConvertToProp = new MultiInstanceToPropositional();

  /** the single-instance weight setting method */
  protected int m_WeightMethod = MultiInstanceToPropositional.WEIGHTMETHOD_1;

  /**
   * Returns a string describing this filter
   *
   * @return a description of the filter suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return
         "A simple Wrapper method for applying standard propositional learners "
       + "to multi-instance data.\n\n"
       + "For more information see:\n\n"
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

    result = new TechnicalInformation(Type.TECHREPORT);
    result.setValue(Field.AUTHOR, "Dánel");
    result.setValue(Field.TITLE, "Lol");
    result.setValue(Field.YEAR, "2016");
    result.setValue(Field.MONTH, "05");
    result.setValue(Field.INSTITUTION, "UCLV");
    result.setValue(Field.ADDRESS, "CEI");

    return result;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector result = new Vector();

    result.addElement(new Option(
          "\tThe type of weight setting for each single-instance:\n"
          + "\t0.keep the weight to be the same as the original value;\n"
          + "\t1.weight = 1.0\n"
          + "\t2.weight = 1.0/Total number of single-instance in the\n"
          + "\t\tcorresponding bag\n"
          + "\t3. weight = Total number of single-instance / (Total\n"
          + "\t\tnumber of bags * Total number of single-instance \n"
          + "\t\tin the corresponding bag).\n"
          + "\t(default: 1)",
          "A", 1, "-A [0|1|2|3]"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      result.addElement(enu.nextElement());
    }

    return result.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   *
   * <pre> -A [0|1|2|3]
   *  The type of weight setting for each single-instance:
   *  0.keep the weight to be the same as the original value;
   *  1.weight = 1.0
   *  2.weight = 1.0/Total number of single-instance in the
   *   corresponding bag
   *  3. weight = Total number of single-instance / (Total
   *   number of bags * Total number of single-instance
   *   in the corresponding bag).
   *  (default: 3)</pre>
   *
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   *
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.rules.ZeroR)</pre>
   *
   * <pre>
   * Options specific to classifier weka.classifiers.rules.ZeroR:
   * </pre>
   *
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   *
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    setDebug(Utils.getFlag('D', options));

    String weightString = Utils.getOption('A', options);
    if (weightString.length() != 0) {
      setWeightMethod(
          new SelectedTag(
            Integer.parseInt(weightString),
            MultiInstanceToPropositional.TAGS_WEIGHTMETHOD));
    } else {
      setWeightMethod(
          new SelectedTag(
            MultiInstanceToPropositional.WEIGHTMETHOD_INVERSE2,
            MultiInstanceToPropositional.TAGS_WEIGHTMETHOD));
    }

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;

    result  = new Vector();

    result.add("-P");

    result.add("-A");
    result.add("" + m_WeightMethod);

    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);

    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String weightMethodTipText() {
    return "The method used for weighting the instances.";
  }

  public void setPositiveLabel(int positiveLabel) {
    this.positiveLabel = positiveLabel;
  }

  /**
   * The new method for weighting the instances.
   *
   * @param method      the new method
   */
  public void setWeightMethod(SelectedTag method){
    if (method.getTags() == MultiInstanceToPropositional.TAGS_WEIGHTMETHOD)
      m_WeightMethod = method.getSelectedTag().getID();
  }

  /**
   * Returns the current weighting method for instances.
   *
   * @return the current weighting method
   */
  public SelectedTag getWeightMethod(){
    return new SelectedTag(
                  m_WeightMethod, MultiInstanceToPropositional.TAGS_WEIGHTMETHOD);
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String methodTipText() {
    return "The method used for testing.";
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    if (super.getCapabilities().handles(Capability.NOMINAL_CLASS))
      result.enable(Capability.NOMINAL_CLASS);
    if (super.getCapabilities().handles(Capability.BINARY_CLASS))
      result.enable(Capability.BINARY_CLASS);
    result.enable(Capability.RELATIONAL_ATTRIBUTES);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // other
    result.enable(Capability.ONLY_MULTIINSTANCE);

    return result;
  }

  /**
   * Returns the capabilities of this multi-instance classifier for the
   * relational data.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getMultiInstanceCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.enable(Capability.NO_CLASS);

    return result;
  }

  /**
   * Builds the classifier
   *
   * @param data the training data to be used for generating the
   * boosted classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    Instances train = new Instances(data);
    train.deleteWithMissingClass();

    if (m_Classifier == null) {
      throw new Exception("A base classifier has not been specified!");
    }

    if (getDebug())
      System.out.println("Start training ...");

    //convert the training dataset into single-instance dataset
    m_ConvertToProp.setWeightMethod(getWeightMethod());
    m_ConvertToProp.setInputFormat(train);
    train = Filter.useFilter(train, m_ConvertToProp);
    train.deleteAttributeAt(0); // remove the bag index attribute

    m_Classifier.buildClassifier(train);
  }

  @Override
  public double classifyInstance(Instance exmp) throws Exception {
    Instances testData = new Instances (exmp.dataset(),0);
    testData.add(exmp);

    // convert the training dataset into single-instance dataset
    m_ConvertToProp.setWeightMethod(
        new SelectedTag(
          MultiInstanceToPropositional.WEIGHTMETHOD_ORIGINAL,
          MultiInstanceToPropositional.TAGS_WEIGHTMETHOD));
    testData = Filter.useFilter(testData, m_ConvertToProp);
    testData.deleteAttributeAt(0); //remove the bag index attribute

    boolean posBag = false;
    for (int i = 0; !posBag && (i < testData.numInstances()); i++) {
      Instance x = testData.get(i);
      double labelIndex = m_Classifier.classifyInstance(x);
      posBag = (labelIndex == positiveLabel);
    }
    return (posBag)?positiveLabel:1-positiveLabel;
  }


  /**
   * Gets a string describing the classifier.
   *
   * @return a string describing the classifer built.
   */
  public String toString() {
    return "MIWrapper with base classifier: \n"+m_Classifier.toString();
  }

  /**
   * Returns the revision string.
   *
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8109 $");
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain the command line arguments to the
   * scheme (see Evaluation)
   */
  public static void main(String[] argv) {
    runClassifier(new MIStdWrapper(), argv);
  }
}
