/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

import weka.core.Utils;
import weka.filters.Filter;

/**
 * UNDER CONSTRUCTION !
 *
 * @author Danel
 */
public class FilterEntry {

  private String name;
  private String command;

  public FilterEntry(String name, String command) {
    this.name = name;
    this.command = command;
  }

  public String name() {
    return name;
  }

  public Filter instantiate() {
    Filter filter = null;
    try {
      String[] cmd = Utils.splitOptions(command);
      String filterFullName = cmd[0];
      cmd[0] = "";
      filter = (Filter)Utils.forName(Class.forName(filterFullName), name, null);
    } catch (Exception e) {
      System.err.println("Classifier construction failed: " + e.getMessage());
    }
    return filter;
  }

}
