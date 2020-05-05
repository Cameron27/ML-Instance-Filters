package weka.filters.unsupervised.instance;

import org.junit.Test;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.Stream;

public class MixupTest {
    @Test
    public void sampleBetaDistribution() {
        double a = 1;
        final long times = 1000000;
        final double target = 2e-2;
        final double binSize = 0.001;
        final Random rnd = new Random(0);

        for (int n = 0; n < 10; n++) {
            int[] bins = new int[(int) (1 / binSize)];
            double[] errors = new double[(int) (1 / binSize)];
            int timesTotal = 0;
            a /= 2;

            while (true) {
                timesTotal += times;

                double finalA = a;
                Stream.generate(() -> Mixup.sampleBetaDistribution(finalA, rnd))
                        .limit(times)
                        .forEach((v) -> {
                            if (v == 1.0) {
                                bins[bins.length - 1]++;
                                return;
                            }
                            bins[(int) (v / binSize)]++;

                        });

                int i = 0;
                while (i * binSize < 1) {
                    double finali = i;
                    double expected = (weka.core.Statistics.incompleteBeta(a, a, (finali + 1) * binSize) - weka.core.Statistics.incompleteBeta(a, a, finali * binSize)) * timesTotal;
                    double actual = bins[i];
                    double error = (expected - actual) / expected;
                    errors[i] = Math.abs(error);
                    i++;
                }

                double maxError = Arrays.stream(errors).max().getAsDouble();
                if (maxError < target) {
                    System.out.println(a + ", " + timesTotal + ": " + maxError);
                    break;
                }
            }
        }
    }
}