/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.Instance;

/**
 *
 * @author Danel
 */
public class GaussianKDE {

  private Instance[] neg;
  private final double h = 0.5; // tunable parameter?
  private int numRecords;
  private Instance[] mostPosInstances;
  private Instance[] mostNegInstances;

  /**
   * Constructs a density estimator object with Gaussian kernel.
   * @param negatives set of negative instances.
   * @param numRecords number of most positive and most negative instance from
   * each bag which can be asked to this object.
   */
  public GaussianKDE(Instance[] negatives, int numRecords) {
    this.neg = negatives;
    this.numRecords = numRecords;
    mostPosInstances = new Instance[numRecords];
    mostNegInstances = new Instance[numRecords];
  }

private void updateMinValues(double[] valArray, double newVal, int[] indexArray, int newIndex) {
  for (int i = 0; i < valArray.length; i++) {
    if (newVal < valArray[i]) {
      if (i < valArray.length - 1) {
        System.arraycopy(valArray, i, valArray, i + 1, valArray.length - i - 1);
        System.arraycopy(indexArray, i, indexArray, i + 1, indexArray.length - i - 1);
      }
      valArray[i] = newVal;
      indexArray[i] = newIndex;
    }
  }
}

private void updateMaxValues(double[] valArray, double newVal, int[] indexArray, int newIndex) {
  for (int i = 0; i < valArray.length; i++) {
    if (newVal > valArray[i]) {
      if (i < valArray.length - 1) {
        System.arraycopy(valArray, i, valArray, i + 1, valArray.length - i - 1);
        System.arraycopy(indexArray, i, indexArray, i + 1, indexArray.length - i - 1);
      }
      valArray[i] = newVal;
      indexArray[i] = newIndex;
    }
  }
}

  public double positiveness(Instance x) {
    double sum = 0;
    for (int i = 0; i < neg.length; i++) {
      double dist2 = Metrics.euclidean2(x, neg[i]);
      double kernel = Math.exp( - dist2 / h);
      sum += kernel;
    }
    double denom = neg.length * Math.pow(h, x.numAttributes());
    double positiveness = sum / denom;
    return positiveness;
  }

  public void analyzeBag(Instance bag) {
    int[] mostPosIndex = new int[numRecords];
    int[] mostNegIndex = new int[numRecords];
    double[] mostPosValues = new double[numRecords];
    double[] mostNegValues = new double[numRecords];
    java.util.Arrays.fill(mostPosValues, 1);
    java.util.Arrays.fill(mostNegValues, 0);
    for (int i = 0; i < bag.relationalValue(1).size(); i++) {
      Instance x = bag.relationalValue(1).get(i);
      double v = positiveness(x);
      updateMinValues(mostPosValues, v, mostPosIndex, i);
      updateMaxValues(mostNegValues, v, mostNegIndex, i);
    }
    for (int i = 0; i < numRecords; i++) {
      mostPosInstances[i] = bag.relationalValue(1).get(mostPosIndex[i]);
      mostNegInstances[i] = bag.relationalValue(1).get(mostNegIndex[i]);
    }
  }

  public Instance getMostPositive(int rank) {
    return mostPosInstances[rank];
  }

  public Instance getMostNegative(int rank) {
    return mostNegInstances[rank];
  }

}
