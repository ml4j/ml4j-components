package org.ml4j.nn.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.base.DifferentiableActivationFunctionTestBase;
import org.ml4j.nn.activationfunctions.mocks.DummyDifferentiableActivationFunctionImpl;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyDifferentiableActivationFunctionImplTest extends DifferentiableActivationFunctionTestBase {

	@Override
	protected MatrixFactory createMatrixFactory() {
		return Mockito.mock(MatrixFactory.class);
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}

	@Override
	protected DifferentiableActivationFunction createDifferentiableActivationFunctionUnderTest(Neurons leftNeurons,
			Neurons rightNeurons) {
		return new DummyDifferentiableActivationFunctionImpl(ActivationFunctionType.createCustomBaseType("DUMMY"), true);
	}

}
