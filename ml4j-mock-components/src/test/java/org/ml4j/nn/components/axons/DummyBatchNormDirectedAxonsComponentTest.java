package org.ml4j.nn.components.axons;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.components.axons.base.BatchNormDirectedAxonsComponentTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyBatchNormDirectedAxonsComponentTest extends BatchNormDirectedAxonsComponentTestBase {
	
	@Override
	protected <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormDirectedAxonsComponentUnderTest(
			ScaleAndShiftAxons<N> axons) {
		return new DummyBatchNormDirectedAxonsComponent<>(axons);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return Mockito.mock(MatrixFactory.class);
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}
}
