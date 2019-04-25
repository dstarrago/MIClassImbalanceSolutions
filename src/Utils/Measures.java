/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.DenseInstance;

/**
 *
 * @author Dánel
 */
public class Measures {

  static public double SparseCos(SparseInstance a, SparseInstance b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numValues(); i++) {
      if (a.index(i) == 0 || a.index(i) == a.classIndex()) continue;    // a.classIndex() ¿=? a.numAttributes() - 1
      a2 += a.valueSparse(i) * a.valueSparse(i);
    }
    for (int i = 0, bi; i < b.numValues(); i++) {
      bi = b.index(i);
      if (bi == 0  || bi == b.classIndex()) continue;
      b2 += b.valueSparse(i) * b.valueSparse(i);
      double av = a.value(bi);
      if (av != 0)
        ab += av * b.valueSparse(i);
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;
  }

  static public double DenseCos(DenseInstance a, DenseInstance b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numAttributes(); i++) {
      ab += a.value(i) * b.value(i);
      a2 += a.value(i) * a.value(i);
      b2 += b.value(i) * b.value(i);
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;
  }

  static public double cos(Instance a, Instance b) {
    if (a instanceof SparseInstance)        // Asumo que b tambien es SparseInstance
      return SparseCos((SparseInstance)a, (SparseInstance)b);
    else
      return DenseCos((DenseInstance)a, (DenseInstance)b);
  }

  static public double cosDistance(Instance a, Instance b) {
    return 1 - cos(a, b);
  }
  
  static public double cos(Instance a, double[] b) {
    if (a instanceof SparseInstance)
      return sparseCos((SparseInstance)a, b);
    else
      return denseCos((DenseInstance)a, b);
  }

  static public double sparseCos(SparseInstance a, double[] b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0, ai; i < a.numValues(); i++) {
      ai = a.index(i);
      a2 += a.valueSparse(i) * a.valueSparse(i);
      b2 += b[ai] * b[ai];
      ab += b[ai] * a.valueSparse(i);
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;
  }

  static public double denseCos(DenseInstance a, double[] b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numAttributes(); i++) {
      ab += a.value(i) * b[i];
      a2 += a.value(i) * a.value(i);
      b2 += b[i] * b[i];
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;
  }

  static public double cos(double[] a, double[] b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.length; i++) {
      ab += a[i] * b[i];
      a2 += a[i] * a[i];
      b2 += b[i] * b[i];
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;
  }

  /**
   * Calcula la distancia mínima entre una instancia y una bolsa. Emplea la
   * distancia coseno entre instancias.
   *
   * @param a instancia
   * @param B bolsa
   * @return
   */
  static private double min(Instance a, Instance B) {
    double m = Double.MAX_VALUE;
    Instances bagB = B.relationalValue(1);
    for (int i = 0; i < bagB.numInstances(); i++) {
      Instance x = bagB.instance(i);
      double s = cosDistance(x, a);   // the cosine distance is the complement of the cosine similarity
      //double s = NEuclidean(x, a);
      if (s < m)
        m = s;
    }
    return m;
  }

  /**
   *
   * @param A primera bolsa a comparar
   * @param B segunda bolsa a comparar
   * @return Número decimal entre 0 y 1. Mientras más cercano a 1, mayor es la
   * distancia.
   */
  static private double max_min(Instance A, Instance B) {
    double m = 0;
    Instances bagA = A.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      double s = min(x, B);
      if (s == 1) return 1;
      if (s > m)
        m = s;
    }
    return m;
  }

  static private double min_min(Instance A, Instance B) {
    double m = Double.MAX_VALUE;
    Instances bagA = A.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      double s = min(x, B);
      if (s < m)
        m = s;
    }
    return m;
  }

  /**
   * Calcula la distancia de Hausdorff entre dos bolsas.
   * Esta distancia está normalizada entre 0 y 1.
   *
   * @param A primera bolsa a comparar
   * @param B segunda bolsa a comparar
   * @return Número decimal entre 0 y 1. Mientras más cercano a 1, mayor es la
   * distancia.
   */
  static public double hausdorff(Instance A, Instance B) {
    double n1 = max_min(A, B);
    if (n1 == 1) return 1;
    double n2 = max_min(B, A);
    if (n1 > n2)
      return n1;
    else
      return n2;
  }

  /**
   * Calcula la distancia de Hausdorff minima entre dos bolsas.
   * Esta distancia está normalizada entre 0 y 1.
   *
   * @param A
   * @param B
   * @return Número decimal entre 0 y 1. Mientras más cercano a 1, mayor es la
   * distancia.
   */
  static public double hausdorff_min(Instance A, Instance B) {
    return min_min(A, B);
  }

  /**
   * Calcula la distancia de Hausdorff promedio entre dos bolsas.
   * Esta distancia está normalizada entre 0 y 1.
   *
   * @param A
   * @param B
   * @return Número decimal entre 0 y 1. Mientras más cercano a 1, mayor es la
   * distancia.
   */
  static public double hausdorff_ave(Instance A, Instance B) {
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

  static private double hausOWA_max_InstToBag(Instance a, Instance B) {
    Instances bagB = B.relationalValue(1);
    int p = bagB.numInstances();  // numero total de objetos que hay que agregar
    double[] weights = new double[p];   // pesos
    double M = p * (p + 1);
    for (int i = 1; i < p + 1; i++) {
      weights[i - 1] = 2 * (p - i + 1) / M;
    }
    int c = 0;    // contador de instancias
    double[] scalares = new double[p];   // distancias a ser ordenadas
    for (int i = 0; i < bagB.numInstances(); i++) {
      Instance x = bagB.instance(i);
      scalares[c++] = cos(x, a);
    }
    java.util.Arrays.sort(scalares);         // Ordena en orden ascendente
    double f = 0;
    for (int j = 0; j < p; j++) {
      f += weights[j] * scalares[p - j - 1];
    }
    return f;
  }

  static private double hausOWA_min_InstToBag(Instance a, Instance B) {
    Instances bagB = B.relationalValue(1);
    int p = bagB.numInstances();  // numero total de objetos que hay que agregar
    double[] weights = new double[p];   // pesos
    double M = p * (p + 1);
    for (int i = 1; i < p + 1; i++) {
      weights[i - 1] = 2 * i / M;
    }
    int c = 0;    // contador de instancias
    double[] scalares = new double[p];   // distancias a ser ordenadas
    for (int i = 0; i < bagB.numInstances(); i++) {
      Instance x = bagB.instance(i);
      scalares[c++] = cos(x, a);
    }
    java.util.Arrays.sort(scalares);         // Ordena en orden ascendente
    double f = 0;
    for (int j = 0; j < p; j++) {
      f += weights[j] * scalares[p - j - 1];
    }
    return f;
  }

  static private double hausOWA_max_BagToBag(Instance A, Instance B) {
    Instances bagA = A.relationalValue(1);
    int p = bagA.numInstances();  // numero total de objetos que hay que agregar
    double[] weights = new double[p];   // pesos
    double M = p * (p + 1);
    for (int i = 1; i < p + 1; i++) {
      weights[i - 1] = 2 * (p - i + 1) / M;
    }
    int c = 0;    // contador de instancias
    double[] scalares = new double[p];   // distancias a ser ordenadas
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      scalares[c++] = hausOWA_min_InstToBag(x, B);
    }
    java.util.Arrays.sort(scalares);         // Ordena en orden ascendente
    double f = 0;
    for (int j = 0; j < p; j++) {
      f += weights[j] * scalares[p - j - 1];
    }
    return f;
  }

  static private double hausOWA_min_BagToBag(Instance A, Instance B) {
    Instances bagA = A.relationalValue(1);
    int p = bagA.numInstances();  // numero total de objetos que hay que agregar
    double[] weights = new double[p];   // pesos
    double M = p * (p + 1);
    for (int i = 1; i < p + 1; i++) {
      weights[i - 1] = 2 * i / M;
    }
    int c = 0;    // contador de instancias
    double[] scalares = new double[p];   // distancias a ser ordenadas
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      scalares[c++] = hausOWA_min_InstToBag(x, B);
    }
    java.util.Arrays.sort(scalares);         // Ordena en orden ascendente
    double f = 0;
    for (int j = 0; j < p; j++) {
      f += weights[j] * scalares[p - j - 1];
    }
    return f;
  }

  /**
   * Calcula la distancia de Hausdorff máxima con operadores OWA entre dos bolsas.
   * Esta distancia está normalizada entre 0 y 1.
   *
   * @param A
   * @param B
   * @return Número decimal entre 0 y 1. Mientras más cercano a 1, mayor es la
   * distancia.
   */
  static public double hausdorff_OWA_max(Instance A, Instance B) {
    double n1 = hausOWA_max_BagToBag(A, B);
    if (n1 == 1) return 1;
    double n2 = hausOWA_max_BagToBag(B, A);
    if (n1 > n2)
      return n1;
    else
      return n2;
  }

  /**
   * Calcula la distancia de Hausdorff mínima con operadores OWA entre dos bolsas.
   * Esta distancia está normalizada entre 0 y 1.
   *
   * @param A
   * @param B
   * @return Número decimal entre 0 y 1. Mientras más cercano a 1, mayor es la
   * distancia.
   */
  static public double hausdorff_OWA_min(Instance A, Instance B) {
    double n1 = hausOWA_min_BagToBag(A, B);
    if (n1 == 1) return 1;
    double n2 = hausOWA_min_BagToBag(B, A);
    if (n1 > n2)
      return n1;
    else
      return n2;
  }

  /**
   * Returns the correlation coefficient of two double vectors.
   *
   * @param a first instance
   * @param b second instance
   * @return the correlation coefficient
   */
  public static final double correlation(Instance a, Instance b) {
    int i;
    double av1 = 0.0, av2 = 0.0, y11 = 0.0, y22 = 0.0, y12 = 0.0, c;
    int n = a.numAttributes() - 2;
    if (n <= 1) {
      return 1.0;
    }
    for (i = 1; i < n - 1; i++) {
      av1 += a.value(i);
      av2 += b.value(i);
    }
    av1 /= (double) n;
    av2 /= (double) n;
    for (i = 1; i < n - 1; i++) {
      y11 += (a.value(i) - av1) * (a.value(i) - av1);
      y22 += (b.value(i) - av2) * (b.value(i) - av2);
      y12 += (a.value(i) - av1) * (b.value(i) - av2);
    }
    if (y11 * y22 == 0.0) {
      c=1.0;
    } else {
      c = y12 / Math.sqrt(Math.abs(y11 * y22));
    }

    return c;
  }

  /**
   * Returns the correlation coefficient of two double vectors.
   *
   * @param a first instance
   * @param w double vector
   * @return the correlation coefficient
   */
  public static final double correlation(Instance a, double[] w) {
    int i;
    double av1 = 0.0, av2 = 0.0, y11 = 0.0, y22 = 0.0, y12 = 0.0, c;
    int n = a.numAttributes() - 2;
    if (n <= 1) {
      return 1.0;
    }
    for (i = 1; i < n - 1; i++) {
      av1 += a.value(i);
      av2 += w[i-1];
    }
    av1 /= (double) n;
    av2 /= (double) n;
    for (i = 1; i < n - 1; i++) {
      y11 += (a.value(i) - av1) * (a.value(i) - av1);
      y22 += (w[i-1] - av2) * (w[i-1] - av2);
      y12 += (a.value(i) - av1) * (w[i-1] - av2);
    }
    if (y11 * y22 == 0.0) {
      c=1.0;
    } else {
      c = y12 / Math.sqrt(Math.abs(y11 * y22));
    }

    return c;
  }

  static public double SparseGaussian1(SparseInstance a, SparseInstance b) {
    double ab = 0, df, df2 = 0;
    for (int i = 0, ai; i < a.numValues(); i++) {
      ai = a.index(i);
      if (ai == 0 || ai == a.classIndex()) continue;
      double bv = b.value(ai);
      if (bv != 0)
        ab += a.valueSparse(i) * bv;
      df = a.valueSparse(i) - bv;
      df2 += df * df;
    }
    for (int i = 0, bi; i < b.numValues(); i++) {
      bi = b.index(i);
      if (bi == 0 || bi == b.classIndex()) continue;
      double av = a.value(bi);
      if (av == 0)
        df2 += b.valueSparse(i) * b.valueSparse(i);
    }
    if (ab == 0)
      return 0;
    double s = Math.exp( - df2/(ab * 10));
    return (s > 1)? 1 : s;
  }

  static public double gaussian1(Instance a, Instance b) {
    if (a instanceof SparseInstance)        // Asumo que b tambien es SparseInstance
      return SparseGaussian1((SparseInstance)a, (SparseInstance)b);

    double ab = 0, df, df2 = 0;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      ab += a.value(i) * b.value(i);
      df = a.value(i) - b.value(i);
      df2 += df * df;
    }
    if (ab == 0)
      return 0;
    double s = Math.exp( - df2/(ab * 10));
    return (s > 1)? 1 : s;
  }

  static public double SparseGaussian2(SparseInstance a, SparseInstance b) {
    double ab = 0, df, df2 = 0;
    for (int i = 0, ai; i < a.numValues(); i++) {
      ai = a.index(i);
      if (ai == 0 || ai == a.classIndex()) continue;
      double bv = b.value(ai);
      if (bv != 0)
        ab += a.valueSparse(i) * bv;
      df = a.valueSparse(i) - bv;
      df2 += df * df;
    }
    for (int i = 0, bi; i < b.numValues(); i++) {
      bi = b.index(i);
      if (bi == 0 || bi == b.classIndex()) continue;
      double av = a.value(bi);
      if (av == 0)
        df2 += b.valueSparse(i) * b.valueSparse(i);
    }
    if (ab == 0)
      return 0;
    double s = Math.exp( -Math.sqrt(df2/ab) / 10);
    return (s > 1)? 1 : s;
  }

  static public double gaussian2(Instance a, Instance b) {
    if (a instanceof SparseInstance)        // Asumo que b tambien es SparseInstance
      return SparseGaussian2((SparseInstance)a, (SparseInstance)b);

    double ab = 0, df, df2 = 0;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      ab += a.value(i) * b.value(i);
      df = a.value(i) - b.value(i);
      df2 += df * df;
    }
    if (ab == 0)
      return 0;
    double s = Math.exp( -Math.sqrt(df2/ab) / 10);
    return (s > 1)? 1 : s;
  }

  static public double SparseGaussian3(SparseInstance a, SparseInstance b) {
    double ab = 0, df, df2 = 0;
    for (int i = 0, ai; i < a.numValues(); i++) {
      ai = a.index(i);
      if (ai == 0 || ai == a.classIndex()) continue;
      double bv = b.value(ai);
      if (bv != 0)
        ab += a.valueSparse(i) * bv;
      df = a.valueSparse(i) - bv;
      df2 += df * df;
    }
    for (int i = 0, bi; i < b.numValues(); i++) {
      bi = b.index(i);
      if (bi == 0 || bi == b.classIndex()) continue;
      double av = a.value(bi);
      if (av == 0)
        df2 += b.valueSparse(i) * b.valueSparse(i);
    }
    if (ab == 0)
      return 0;
    double s = Math.exp( - df2 / (Math.sqrt(ab) * 10));
    return (s > 1)? 1 : s;
  }

  static public double gaussian3(Instance a, Instance b) {
    if (a instanceof SparseInstance)        // Asumo que b tambien es SparseInstance
      return SparseGaussian3((SparseInstance)a, (SparseInstance)b);

    double ab = 0, df, df2 = 0;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      ab += a.value(i) * b.value(i);
      df = a.value(i) - b.value(i);
      df2 += df * df;
    }
    if (ab == 0)
      return 0;
    double s = Math.exp( - df2 / (Math.sqrt(ab) * 10));
    return (s > 1)? 1 : s;
  }

  static public double cosExp(Instance a, Instance b) {
    double ab = 0;
    double a2 = 0, b2 = 0;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      ab += Math.exp(-a.value(i) - b.value(i));
      a2 += Math.exp(-2 * a.value(i));
      b2 += Math.exp(-2 * b.value(i));
    }
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));
    return (s > 1)? 1 : s;
  }

  static public double euclidean(Instance a, Instance b) {
    double s = 0;
    double dif;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      dif = a.value(i) - b.value(i);
      s += dif * dif;
    }
    return Math.sqrt(s);
  }

  /**
   * Normalized Euclidean distance.
   * Calcula la distancia Euclideana normalizada entre dos instancias.
   * @param a
   * @param b
   * @return Numero decimal entre 0 y 1.
   */
  static public double NEuclidean(Instance a, Instance b) {
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

  /**
   * Euclidean distance in the range [0, 1].
   *
   * @param a
   * @param b
   * @return
   */
  static public double euclideanExp(Instance a, Instance b) {
    return 1 - Math.exp( -euclidean(a, b));
  }

  static public int hierarchial(double[] lowerAndBag, double[] upperAndBag) {
    int length = lowerAndBag.length;
    // Rule 1: if intLowerApp[i] > intUpperApp[j], for all j != i then class <- i

    for (int i = 0; i < length; i++) {
      boolean fail =  false;
      for (int j = 0; j < length; j++) {
        if (j != i)
          if (lowerAndBag[i] <= upperAndBag[j])
            fail = true;
      }
      if (!fail) return i;
    }

    // Rule 2: if intLowerApp[i] > intLowerApp[j], for all j != i then class <- i

    for (int i = 0; i < length; i++) {
      boolean fail =  false;
      for (int j = 0; j < length; j++) {
        if (j != i)
          if (lowerAndBag[i] <= lowerAndBag[j])
            fail = true;
      }
      if (!fail) return i;
    }

    // Rule 3: if intUpperApp[i] > intUpperApp[j], for all j != i then class <- i

    for (int i = 0; i < length; i++) {
      boolean fail =  false;
      for (int j = 0; j < length; j++) {
        if (j != i)
          if (upperAndBag[i] <= upperAndBag[j])
            fail = true;
      }
      if (!fail) return i;
    }
    return -1;
  }

}
