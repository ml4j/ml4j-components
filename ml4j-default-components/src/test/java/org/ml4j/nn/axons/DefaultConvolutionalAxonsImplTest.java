package org.ml4j.nn.axons;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.base.Axons3DTestBase;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.mockito.Mock;
import org.mockito.Mockito;

public class DefaultConvolutionalAxonsImplTest extends Axons3DTestBase<ConvolutionalAxons> {

	@Mock
	private AxonsFactory mockAxonsFactory;

	@Mock
	private FullyConnectedAxons mockFullyConnectedAxons;
	
	@Mock
	private AxonsActivation mockAxonsActivation;

	@Before
	@Override
	public void setUp() {
		super.setUp();
	}

	@Override
	protected ConvolutionalAxons createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config) {
		Mockito.when(mockAxonsFactory.createFullyConnectedAxons(Mockito.any(), Mockito.any()
				,Mockito.any(), Mockito.any())).thenReturn(mockFullyConnectedAxons);
		return new DefaultConvolutionalAxonsImpl(mockAxonsFactory, leftNeurons, rightNeurons, config, null, null);
	}

	@Override
	public void testPushLeftToRight() {
		
		NeuronsActivation nestedOutput = createNeuronsActivation(2, 32 * 400);
		
		Mockito.when(mockAxonsActivation.getPostDropoutOutput()).thenReturn(nestedOutput);
		
		Mockito.when(mockFullyConnectedAxons.pushLeftToRight(Mockito.any(), Mockito.any(), Mockito.any())).thenReturn(mockAxonsActivation);

		super.testPushLeftToRight();
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new NeuronsActivationImpl(matrixFactory.createMatrix(featureCount, exampleCount),
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	}

	@Override
	protected int getExpectedReformattedInputColumns() {
		return 32 * 400;
	}

	@Override
	protected int getExpectedReformattedInputRows() {
		return 81 * 3;
	}

}
