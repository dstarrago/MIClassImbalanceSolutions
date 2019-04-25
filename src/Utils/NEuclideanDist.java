/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.Instance;

/**
 * Class to calculate normalized euclidean distance between instances.
 *
 * @author Danel
 */
public class NEuclideanDist extends InstanceLevelDistance {

  /**
   * Calculate the normalized euclidean distance between two instance.
   * 
   * @param a one instance
   * @param b the other instance
   * @return the distance value
   */
  public double measure(Instance a, Instance b) {
    double s = 0;
    double dif, a2 = 0, b2 = 0;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      dif = a.value(i) - b.value(i);
      s += dif * dif;
      a2 += a.value(i) * a.value(i);
      b2 += b.value(i) * b.value(i);
    }
    double D = Math.sqrt(a2) + Math.sqrt(b2);
    return Math.sqrt(s) / D;
  }

}
