package org.ml4j.nn.axons;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.base.Axons3DTestBase;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

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
		MockitoAnnotations.initMocks(this);
		Mockito.when(leftNeurons.getNeuronCountExcludingBias()).thenReturn(784 * 3);
		Mockito.when(leftNeurons.getDepth()).thenReturn(3);
		Mockito.when(leftNeurons.getWidth()).thenReturn(28);
		Mockito.when(leftNeurons.getHeight()).thenReturn(28);
		Mockito.when(rightNeurons.getNeuronCountExcludingBias()).thenReturn(400 * 2);
		Mockito.when(rightNeurons.getDepth()).thenReturn(2);
		Mockito.when(rightNeurons.getWidth()).thenReturn(20);
		Mockito.when(rightNeurons.getHeight()).thenReturn(20);
		super.setUp();
		Mockito.when(mockAxonsContext.getMatrixFactory()).thenReturn(matrixFactory);
		this.mockLeftToRightInputActivation = createNeuronsActivation(784 * 3, 32);
	}

	@Override
	protected ConvolutionalAxons createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config) {
		config.setFilterWidthAndHeight(leftNeurons, rightNeurons);

		Mockito.when(
				mockAxonsFactory.createFullyConnectedAxons(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any()))
				.thenReturn(mockFullyConnectedAxons);
		return new DefaultConvolutionalAxonsImpl(mockAxonsFactory, leftNeurons, rightNeurons, config, null, null);
	}

	@Override
	public void testPushLeftToRight() {

		NeuronsActivation nestedOutput = createNeuronsActivation(2, 32 * 400);

		Mockito.when(mockAxonsActivation.getPostDropoutOutput()).thenReturn(nestedOutput);

		Mockito.when(mockFullyConnectedAxons.pushLeftToRight(Mockito.any(), Mockito.any(), Mockito.any()))
				.thenReturn(mockAxonsActivation);

		super.testPushLeftToRight();
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new NeuronsActivationImpl(new Neurons(featureCount, false),
				matrixFactory.createMatrix(featureCount, exampleCount),
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT);
	}

	@Override
	protected int getExpectedReformattedInputColumns() {
		return 32 * 400;
	}

	@Override
	protected int getExpectedReformattedInputRows() {
		return 81 * 3;
	}

	@Override
	protected boolean expectPostDropoutInputToBeSet() {
		return true;
	}

}
