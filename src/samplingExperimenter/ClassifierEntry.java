/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Utils;

/**
 *
 * @author Danel
 */
public class ClassifierEntry {

  private String name;
  private String command;

  public ClassifierEntry(String name, String command) {
    this.name = name;
    this.command = command;
  }

  public String name() {
    return name;
  }

  public Classifier instantiate() {
    Classifier scheme = null;
    try {
      String[] cmd = Utils.splitOptions(command);
      String schemeFullName = cmd[0];
      cmd[0] = "";
      scheme = AbstractClassifier.forName(schemeFullName, cmd);
    } catch (Exception e) {
      System.err.println("Classifier construction failed: " + e.getMessage());
    }
    return scheme;
  }

}
