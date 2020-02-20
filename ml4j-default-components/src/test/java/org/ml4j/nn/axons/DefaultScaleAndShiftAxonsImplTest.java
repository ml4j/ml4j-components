package org.ml4j.nn.axons;

import java.util.Arrays;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.base.AxonsTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.mockito.Mockito;

public class DefaultScaleAndShiftAxonsImplTest extends AxonsTestBase<ScaleAndShiftAxons<?>> {

	@Before
	@Override
	public void setUp() {
		super.setUp();
	}

	@Override
	protected ScaleAndShiftAxons<?> createAxonsUnderTest(Neurons leftNeurons, Neurons rightNeurons) {
		AxonWeights axonWeights = new ScaleAndShiftAxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(),
				rightNeurons.getNeuronCountExcludingBias(),
				new WeightsMatrixImpl(matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), 1),
						new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE), 
								WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)),
				new BiasVectorImpl(matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), 1), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT), null);
		return new DefaultScaleAndShiftAxonsImpl<>(new AxonsConfig<>(leftNeurons, rightNeurons), axonWeights);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new NeuronsActivationImpl(new Neurons(featureCount, false),
				matrixFactory.createMatrix(featureCount, exampleCount),
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	}

	@Override
	public void testPushLeftToRight() {
		Mockito.when(mockAxonsContext.getLeftHandInputDropoutKeepProbability()).thenReturn(1f);
		super.testPushLeftToRight();
	}

}
