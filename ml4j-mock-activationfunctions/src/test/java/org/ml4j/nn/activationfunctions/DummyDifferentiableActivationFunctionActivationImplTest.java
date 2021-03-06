package org.ml4j.nn.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.base.DifferentiableActivationFunctionActivationTestBase;
import org.ml4j.nn.activationfunctions.mocks.DummyDifferentiableActivationFunctionActivationImpl;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyDifferentiableActivationFunctionActivationImplTest
		extends DifferentiableActivationFunctionActivationTestBase {

	@Override
	protected DifferentiableActivationFunctionActivation createDifferentiableActivationFunctionActivationUnderTest(
			DifferentiableActivationFunction activationFunction, NeuronsActivation input, NeuronsActivation output) {
		return new DummyDifferentiableActivationFunctionActivationImpl(activationFunction, input, output);
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
