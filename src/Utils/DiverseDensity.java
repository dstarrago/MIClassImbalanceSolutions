/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import java.math.BigDecimal;
import java.math.MathContext;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class that calculates instance's diverse density
 *
 * @author Danel
 */
public class DiverseDensity {

  private Instances data;
  private int posClassLabel;
  private CosineDist cosDist;

  public DiverseDensity(Instances data, int posClassLabel) {
    this.data = data;
    this.posClassLabel = posClassLabel;
    cosDist = new CosineDist();
  }

  /**
   * Calculates the probability that an instance x belongs to the concept t with
   * a Gaussian-like distribution.
   * @param x instance to calculate probability.
   * @param t instance representing the target concept.
   * @return the probability modeled by Gaussian-like distribution.
   */
  protected double instToClassProb(Instance x, Instance t) {
    double dist = cosDist.measure(x, t);
    double dist2 = dist * dist;
    return Math.exp( -dist2 );
  }

  public BigDecimal bigNoisyOr(Instance x) {
    double y;                                                   // the class label
    BigDecimal DD = new BigDecimal(1, MathContext.DECIMAL32);   // the diverse density value
    for (int i = 0; i < data.numInstances(); i++) {
      Instance bag = data.instance(i);
      if (bag.classValue() == posClassLabel) {
          y = 1;
      } else {
          y = -1;
      }
      // a = 0 for negative bags and a = 1 for negative ones
      BigDecimal bdA = new BigDecimal((1 + y) / 2, MathContext.DECIMAL32);
      BigDecimal bdY = new BigDecimal(y, MathContext.DECIMAL32);
      BigDecimal bagProd = new BigDecimal(1, MathContext.DECIMAL32);
      for (int j = 0; j < bag.relationalValue(1).numInstances(); j++) {
        Instance xj = bag.relationalValue(1).get(j);
        BigDecimal mult = new BigDecimal(1 - instToClassProb(x, xj), MathContext.DECIMAL32);
        bagProd = bagProd.multiply(mult, MathContext.DECIMAL32);    // Efficiency issue?
      }
      // DD *= (a - y * bagProd);
      DD = DD.multiply(bdA.subtract(bdY.multiply(bagProd, MathContext.DECIMAL32),
              MathContext.DECIMAL32), MathContext.DECIMAL32);
    }
    //System.out.println(DD);
    return DD;
  }

  /**
   * Evaluates the Most-Likely-Cause model of diverse density. The higher the
   * value of the function, the greater the probability that the instance is positive.
   * @param x an instance inside a bag.
   * @return the diverse density value according to the Most-Likely-Cause model.
   */
  public double mostLikelyCause(Instance x) {
    double y;                                                   // the class label
    double DD = 1;   // the diverse density value
    for (int i = 0; i < data.numInstances(); i++) {
      Instance bag = data.instance(i);
      if (bag.classValue() == posClassLabel) {
          y = 1;
      } else {
          y = -1;
      }
      // a = 1 for negative bags and a = 0 for positive ones
      double a = (1 - y) / 2;
      double maxProb = 0;
      for (int j = 0; j < bag.relationalValue(1).numInstances(); j++) {
        Instance xj = bag.relationalValue(1).get(j);
        double prob = instToClassProb(x, xj);
        if (prob > maxProb)
          maxProb = prob;
      }
      DD *= (a + y * maxProb);
    }
//    System.out.println(DD);
    return DD;
  }

  /**
   * Evaluates the Most-Likely-Cause model of diverse density. The higher the
   * value of the function, the greater the probability that the instance is positive.
   * @param x an instance inside a bag.
   * @return the diverse density value according to the Most-Likely-Cause model.
   */
  public double mostLikelyCauseOWA(Instance x) {
    double y;                                                   // the class label
    double DD = 1;   // the diverse density value
    for (int i = 0; i < data.numInstances(); i++) {
      Instance bag = data.instance(i);
      if (bag.classValue() == posClassLabel) {
          y = 1;
      } else {
          y = -1;
      }
      // a = 1 for negative bags and a = 0 for positive ones
      double a = (1 - y) / 2;
      int bagSize = bag.relationalValue(1).numInstances();
      double[] probs = new double[bagSize];
      OWA owa = new OWA(OWA.vw_LINEAL_MAX, bagSize);
      for (int j = 0; j < bagSize; j++) {
        Instance xj = bag.relationalValue(1).get(j);
        probs[j] = instToClassProb(x, xj);
      }
      double owaProb = owa.operate(probs);
      DD *= (a + y * owaProb);
    }
//    System.out.println(DD);
    return DD;
  }

}
