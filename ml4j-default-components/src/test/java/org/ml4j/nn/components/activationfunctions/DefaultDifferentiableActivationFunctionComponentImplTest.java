package org.ml4j.nn.components.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;

public class DefaultDifferentiableActivationFunctionComponentImplTest
		extends DifferentiableActivationFunctionComponentTestBase<DifferentiableActivationFunctionComponentAdapter> {

	@Mock
	protected DifferentiableActivationFunction mockActivationFunction;

	@Override
	protected DifferentiableActivationFunctionComponentAdapter createDifferentiableActivationFunctionComponentUnderTest(
			Neurons neurons, ActivationFunctionType activationFunctionType) {

		Mockito.when(mockActivationFunction.getActivationFunctionType())
				.thenReturn(ActivationFunctionType.createCustomBaseType("DUMMY"));

		Mockito.when(mockActivationFunction.activate(mockNeuronsActivation, mockNeuronsActivationContext))
				.thenReturn(mockActivationFunctionActivation);

		return new DefaultDifferentiableActivationFunctionComponentImpl("someName", neurons, mockActivationFunction);
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
