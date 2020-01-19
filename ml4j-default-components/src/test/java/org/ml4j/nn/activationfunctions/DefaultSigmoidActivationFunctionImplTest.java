package org.ml4j.nn.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.activationfunctions.base.DifferentiableActivationFunctionTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons1D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DefaultSigmoidActivationFunctionImplTest extends DifferentiableActivationFunctionTestBase {

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new NeuronsActivationImpl(new Neurons1D(featureCount, false), matrixFactory.createMatrix(featureCount, exampleCount),
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	}

	@Override
	protected DifferentiableActivationFunction createDifferentiableActivationFunctionUnderTest(Neurons leftNeurons,
			Neurons rightNeurons) {
		return new DefaultSigmoidActivationFunctionImpl();
	}

}
