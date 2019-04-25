/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.Instance;

/**
 * Abstract class to be extended by all classes that calculate distance between bags.
 *
 * @author Danel
 */
public abstract class BagLevelDistance {

  private InstanceLevelDistance instDistance;

  public BagLevelDistance(InstanceLevelDistance instDistance) {
    this.instDistance = instDistance;
  }

  /**
   * Calculate a distance measure between two bags.
   *
   * @param A one bag
   * @param B the other bag
   * @return the distance value
   */
  public abstract double measure(Instance A, Instance B);

  protected InstanceLevelDistance instDistance() {
    return instDistance;
  }

}
