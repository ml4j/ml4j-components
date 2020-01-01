package org.ml4j.nn.components.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DefaultDifferentiableActivationFunctionComponentImplTest extends DifferentiableActivationFunctionComponentTestBase {

	@Override
	protected DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponentUnderTest(Neurons neurons, 
			DifferentiableActivationFunction activationFunction) {
		return new DefaultDifferentiableActivationFunctionComponentImpl(neurons, activationFunction);
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
