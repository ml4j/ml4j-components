package org.ml4j.nn.axons;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.base.AxonsTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
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
				matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), 1),
				matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), 1), null);
		return new DefaultScaleAndShiftAxonsImpl<>(leftNeurons, rightNeurons, axonWeights);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new NeuronsActivationImpl(new Neurons(featureCount, false),
				matrixFactory.createMatrix(featureCount, exampleCount),
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	}

	@Override
	public void testPushLeftToRight() {
		Mockito.when(mockAxonsContext.getLeftHandInputDropoutKeepProbability()).thenReturn(1f);
		super.testPushLeftToRight();
	}

}
