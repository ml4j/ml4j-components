package org.ml4j.nn.components.onetoone;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphActivationTestBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyDirectedComponentChainBipoleGraphActivationTest extends DefaultDirectedComponentChainBipoleGraphActivationTestBase {

	@Override
	protected DefaultDirectedComponentChainBipoleGraphActivation createDefaultDirectedComponentChainBipoleGraphActivationUnderTest(
			DefaultDirectedComponentChainBipoleGraph bipoleGraph, NeuronsActivation output) {
		return new DummyDefaultDirectedComponentChainBipoleGraphActivation(bipoleGraph, output);
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
