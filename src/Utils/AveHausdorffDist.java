/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Class to calculate average Hausdorff distance between bags.
 *
 * @author Danel
 */
public class AveHausdorffDist extends BagLevelDistance {

  public AveHausdorffDist(InstanceLevelDistance instDistance) {
    super(instDistance);
  }

  /**
   * Calculate the average Hausdorff distance between two bags.
   *
   * @param A one bag
   * @param B the other bag
   * @return the distance value
   */
  public double measure(Instance A, Instance B) {
    double m = 0;
    Instances bagA = A.relationalValue(1);
    Instances bagB = B.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      m += min(x, B);
    }
    for (int i = 0; i < bagB.numInstances(); i++) {
      Instance x = bagB.instance(i);
      m += min(x, A);
    }
    int N = bagA.numInstances() + bagB.numInstances();
    m /= N;
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
