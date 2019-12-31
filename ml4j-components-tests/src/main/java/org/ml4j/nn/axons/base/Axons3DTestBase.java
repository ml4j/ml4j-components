package org.ml4j.nn.axons.base;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
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
		axons = createAxonsUnderTest(leftNeurons, rightNeurons, 0, 0, 0, 0);
		this.mockLeftToRightInputActivation = createNeuronsActivation(100, 32);
	}
	
	protected abstract A createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight);
	
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
		NeuronsActivation postDropoutInput = leftToRightActivation.getPostDropoutInput();
		Assert.assertNotNull(postDropoutInput);
		Assert.assertEquals(mockLeftToRightInputActivation.getFeatureCount(), postDropoutInput.getFeatureCount());
		Assert.assertEquals(mockLeftToRightInputActivation.getExampleCount(), postDropoutInput.getExampleCount());
		Assert.assertEquals(mockLeftToRightInputActivation.getFeatureOrientation(), postDropoutInput.getFeatureOrientation());

		NeuronsActivation postDropoutOutput = leftToRightActivation.getPostDropoutOutput();
		Assert.assertNotNull(postDropoutOutput);

		
	}
	
}
