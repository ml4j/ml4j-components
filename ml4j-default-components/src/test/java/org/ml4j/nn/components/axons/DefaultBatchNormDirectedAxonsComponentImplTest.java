package org.ml4j.nn.components.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.components.axons.base.BatchNormDirectedAxonsComponentTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.mockito.Mock;
import org.mockito.Mockito;

public class DefaultBatchNormDirectedAxonsComponentImplTest extends BatchNormDirectedAxonsComponentTestBase {

	@Mock
	protected AxonsActivation mockAxonsActivation;

	@Mock
	protected NeuronsActivation mockOutputActivation;

	@Override
	protected <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormDirectedAxonsComponentUnderTest(
			ScaleAndShiftAxons<N> axons) {
		Matrix mean = null;
		Matrix stddev = null;
		return new DefaultBatchNormDirectedAxonsComponentImpl<>("someName", axons, mean, stddev);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new NeuronsActivationImpl(new Neurons(featureCount, false),
				matrixFactory.createMatrix(featureCount, exampleCount),
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	}

	@Override
	public void testForwardPropagate() {
		Mockito.when(mockOutputActivation.getFeatureCount()).thenReturn(120);
		Mockito.when(mockOutputActivation.getExampleCount()).thenReturn(32);
		Mockito.when(mockOutputActivation.getFeatureOrientation())
				.thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Mockito.when(mockAxonsActivation.getPostDropoutOutput()).thenReturn(mockOutputActivation);

		// TODO verify
		Mockito.when(mockAxons.pushLeftToRight(Mockito.any(), Mockito.isNull(AxonsActivation.class),
				Mockito.eq(mockAxonsContext))).thenReturn(mockAxonsActivation);

		Mockito.when(mockAxonsContext.getMatrixFactory()).thenReturn(matrixFactory);

		super.testForwardPropagate();
	}
}
