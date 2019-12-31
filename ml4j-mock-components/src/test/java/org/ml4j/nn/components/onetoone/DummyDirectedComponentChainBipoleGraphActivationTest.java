package org.ml4j.nn.components.onetoone;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphActivationTestBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyDirectedComponentChainBipoleGraphActivationTest extends DefaultDirectedComponentChainBipoleGraphActivationTestBase {

	@Override
	protected DefaultDirectedComponentBipoleGraphActivation createDefaultDirectedComponentChainBipoleGraphActivationUnderTest(
			DefaultDirectedComponentBipoleGraph bipoleGraph, NeuronsActivation output) {
		return new DummyDefaultDirectedComponentBipoleGraphActivation(bipoleGraph, output);
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
