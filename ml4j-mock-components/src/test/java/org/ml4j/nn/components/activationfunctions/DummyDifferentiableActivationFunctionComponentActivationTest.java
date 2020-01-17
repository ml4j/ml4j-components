package org.ml4j.nn.components.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentActivationTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;

public class DummyDifferentiableActivationFunctionComponentActivationTest extends DifferentiableActivationFunctionComponentActivationTestBase<DifferentiableActivationFunctionComponent> {

	@Mock
	private DifferentiableActivationFunctionComponent mock;
	
	@Override
	protected DifferentiableActivationFunctionComponentActivation createDifferentiableActivationFunctionComponentActivationUnderTest(
			DifferentiableActivationFunctionComponent activationFunction, NeuronsActivation input, NeuronsActivation output) {
		return new DummyDifferentiableActivationFunctionComponentActivation(activationFunction, input, output);
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
	protected DifferentiableActivationFunctionComponent createMockActivationFunctionComponent() {
		return mock;
	}
}
