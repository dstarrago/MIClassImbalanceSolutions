/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MultiInstanceToPropositional;
import weka.core.Capabilities.Capability;

/**
 * Multi-Instance classifier based in propositionalization of MI data and
 * standard MI assumption application.
 *
 * @author Danel
 */
public class PropMIStandard extends SingleClassifierEnhancer {

  private Instances proposData;

  private double positiveLabel = 1;

  public void buildClassifier(Instances data) throws Exception {
    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();

    if (m_Classifier == null)
      throw new Exception("Classifier undefined");

    MultiInstanceToPropositional filter = new MultiInstanceToPropositional();
    filter.setInputFormat(data);
    proposData = Filter.useFilter(data, filter);
    proposData.deleteAttributeAt(0); // delete the bagID attribute
    proposData.setClassIndex(proposData.numAttributes()-1);

    m_Classifier.buildClassifier(proposData);

  }

  public void setPositiveLabel(double positiveLabel) {
    this.positiveLabel = positiveLabel;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.RELATIONAL_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    if (super.getCapabilities().handles(Capability.NOMINAL_CLASS))
      result.enable(Capability.NOMINAL_CLASS);
    if (super.getCapabilities().handles(Capability.BINARY_CLASS))
      result.enable(Capability.BINARY_CLASS);
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

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.disableAllClasses();
    result.enable(Capability.NO_CLASS);

    return result;
  }

  @Override
  public double classifyInstance(Instance instance) throws Exception {
    Instances bag = instance.relationalValue(1);
    boolean posBag = false;
    for (int i = 0; !posBag && (i < bag.numInstances()); i++) {
      Instance x = bag.get(i);
      double labelIndex = m_Classifier.classifyInstance(x);
      posBag = (labelIndex == positiveLabel);
    }
    return (posBag)?positiveLabel:1-positiveLabel;
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain the command line arguments to the
   * scheme (see Evaluation)
   */
  public static void main(String[] argv) {
    runClassifier(new PropMIStandard(), argv);
  }
}
