package org.ml4j.nn.axons.base;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class AxonsTestBase<A extends Axons<?, ?, ?>> {

	private A axons;
	
	@Mock
	private Neurons leftNeurons;
	
	@Mock
	private Neurons rightNeurons;
	
	@Mock
	private AxonsContext mockAxonsContext;
	
	@Mock
	private NeuronsActivation mockLeftToRightInputActivation;

	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		axons = createAxonsUnderTest(leftNeurons, rightNeurons);
	}
	
	protected abstract A createAxonsUnderTest(Neurons leftNeurons, Neurons rightNeurons);
	
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
		
		Mockito.when(mockLeftToRightInputActivation.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		Mockito.when(mockLeftToRightInputActivation.getExampleCount()).thenReturn(32);
		Mockito.when(mockLeftToRightInputActivation.getFeatureCount()).thenReturn(100);

		
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
