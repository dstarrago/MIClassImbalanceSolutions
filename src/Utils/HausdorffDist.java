/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.Instance;

/**
 * Class to calculate standard Hausdorff distance between bags.
 *
 * @author Danel
 */
public class HausdorffDist extends BagLevelDistance {

  public HausdorffDist(InstanceLevelDistance instDistance) {
    super(instDistance);
  }

  /**
   * Calculate the standard Hausdorff distance between two bags.
   *
   * @param A one bag
   * @param B the other bag
   * @return the distance value
   */
  public double measure(Instance A, Instance B) {
    double n1 = max_min(A, B);
    if (n1 == 1) return 1;
    double n2 = max_min(B, A);
    if (n1 > n2)
      return n1;
    else
      return n2;
  }

  private double max_min(Instance A, Instance B) {
    double m = 0;
    for (int i = 0; i < A.relationalValue(1).numInstances(); i++) {
      Instance x = A.relationalValue(1).instance(i);
      double s = min(x, B);
      if (s == 1) return 1;
      if (s > m)
        m = s;
    }
    return m;
  }

  private double min(Instance a, Instance B) {
    double m = Double.MAX_VALUE;
    for (int i = 0; i < B.relationalValue(1).numInstances(); i++) {
      Instance x = B.relationalValue(1).instance(i);
      double s = instDistance().measure(a, x);
      if (s < m)
        m = s;
    }
    return m;
  }

}
