package weka.filters.unsupervised.instance;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

public class MixupTest {
    @Test
    public void sampleBetaDistribution() {
        final double a = 0.5;
        final int times = 1000000;
        List<Double> results = new ArrayList<>();
        Random rnd = new Random(0);

        // make samples
        for (int i = 0; i < times; i++) {
            results.add(Mixup.sampleBetaDistribution(a, rnd));
        }

        double step = 0.01;
        int i = 0;
        List<Double> diffs = new ArrayList<>();

        // calculate difference between expected count and actual count in each step size
        while (i * step < 1) {
            double finali = i;
            double expected = weka.core.Statistics.incompleteBeta(a, a, (finali + 1) * step) - weka.core.Statistics.incompleteBeta(a, a, finali * step);

            diffs.add(results.stream().filter(d -> d >= finali * step && d < (finali + 1) * step).count() / (double) times - expected);

            i++;
        }

        // checks expected vs sample is sufficiently small
        assertTrue(diffs.stream().noneMatch(d -> d >= 1e-3));
    }
}