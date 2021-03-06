package org.ml4j.nn.components.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyDifferentiableActivationFunctionComponentTest
		extends DifferentiableActivationFunctionComponentTestBase<DifferentiableActivationFunctionComponent> {

	@Override
	protected DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponentUnderTest(
			Neurons neurons, ActivationFunctionType activationFunctionType) {
		return new DummyDifferentiableActivationFunctionComponent("someName", neurons, activationFunctionType);
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
