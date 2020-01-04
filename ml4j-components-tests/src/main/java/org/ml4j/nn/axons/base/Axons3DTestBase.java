package org.ml4j.nn.axons.base;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class Axons3DTestBase<A extends Axons<?, ?, ?>> extends TestBase {

	private A axons;
	
	@Mock
	private Neurons3D leftNeurons;
	
	@Mock
	private Neurons3D rightNeurons;
	
	@Mock
	private AxonsContext mockAxonsContext;
	
	private NeuronsActivation mockLeftToRightInputActivation;

	@Before
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
		axons = createAxonsUnderTest(leftNeurons, rightNeurons, new Axons3DConfig());
		Mockito.when(mockAxonsContext.getMatrixFactory()).thenReturn(matrixFactory);
		this.mockLeftToRightInputActivation = createNeuronsActivation(784 * 3, 32);
	}
	
	protected abstract A createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config);
	
	protected abstract int getExpectedReformattedInputRows();
	protected abstract int getExpectedReformattedInputColumns();

	
	@Test
	public void testGetLeftNeurons() {
		Neurons leftNeurons = axons.getLeftNeurons();
		Assert.assertNotNull(leftNeurons);
	}
	
	@Test
	public void testGetRightNeurons() {
		Neurons leftNeurons = axons.getRightNeurons();
		Assert.assertNotNull(leftNeurons);
	}
	
	@Test
	public void testPushLeftToRight() {
		
		AxonsActivation leftToRightActivation = axons.pushLeftToRight(mockLeftToRightInputActivation, null, mockAxonsContext);
		Assert.assertNotNull(leftToRightActivation);
		NeuronsActivation postDropoutInput = leftToRightActivation.getPostDropoutInput().get();
		Assert.assertNotNull(postDropoutInput);
		Assert.assertEquals(getExpectedReformattedInputRows(), postDropoutInput.getFeatureCount());
		Assert.assertEquals(getExpectedReformattedInputColumns(), postDropoutInput.getExampleCount());
		Assert.assertEquals(mockLeftToRightInputActivation.getFeatureOrientation(), postDropoutInput.getFeatureOrientation());

		NeuronsActivation postDropoutOutput = leftToRightActivation.getPostDropoutOutput();
		Assert.assertNotNull(postDropoutOutput);

		
	}
	
}
