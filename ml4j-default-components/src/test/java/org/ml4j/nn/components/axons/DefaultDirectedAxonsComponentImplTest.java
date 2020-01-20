package org.ml4j.nn.components.axons;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;

public class DefaultDirectedAxonsComponentImplTest extends DirectedAxonsComponentTestBase {

	@Mock
	private AxonsActivation mockAxonsActivation;

	@Override
	protected <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponentUnderTest(
			Axons<L, R, ?> axons) {
		return new DefaultDirectedAxonsComponentImpl<>(axons);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return Mockito.mock(MatrixFactory.class);
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}

	@Override
	public void testForwardPropagate() {
		Mockito.when(mockAxonsActivation.getPostDropoutOutput()).thenReturn(mockOutputActivation);
		Mockito.when(mockAxons.pushLeftToRight(mockInputActivation, null, mockAxonsContext))
				.thenReturn(mockAxonsActivation);
		super.testForwardPropagate();
	}
}
