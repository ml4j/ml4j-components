package org.ml4j.nn.components.axons;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentActivationTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DefaultDirectedAxonsComponentActivationImplTest extends DirectedAxonsComponentActivationTestBase {

	@Override
	protected <L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> DirectedAxonsComponentActivation createDirectedAxonsComponentActivationUnderTest(
			DirectedAxonsComponent<L, R, A> axonsComponent, AxonsActivation axonsActivation,
			AxonsContext axonsContext) {
		return new DefaultDirectedAxonsComponentActivationImpl<>(axonsComponent, axonsActivation, axonsContext);
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
