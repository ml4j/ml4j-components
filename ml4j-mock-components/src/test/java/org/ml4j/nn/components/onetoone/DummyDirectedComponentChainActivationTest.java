package org.ml4j.nn.components.onetoone;

import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainActivationTestBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyDirectedComponentChainActivationTest extends DefaultDirectedComponentChainActivationTestBase {

	@Override
	protected DefaultDirectedComponentChainActivation createDefaultDirectedComponentChainActivationUnderTest(
			DefaultDirectedComponentChain componentChain,
			List<DefaultChainableDirectedComponentActivation> activations) {
		return new DummyDefaultDirectedComponentActivation(componentChain, activations);
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
