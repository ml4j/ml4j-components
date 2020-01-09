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
	protected Neurons3D leftNeurons;
	
	@Mock
	protected Neurons3D rightNeurons;
	
	@Mock
	protected AxonsContext mockAxonsContext;
	
	protected NeuronsActivation mockLeftToRightInputActivation;

	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		axons = createAxonsUnderTest(leftNeurons, rightNeurons, new Axons3DConfig());
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
	
	protected abstract boolean expectPostDropoutInputToBeSet();
	
	@Test
	public void testPushLeftToRight() {

		Mockito.when(mockAxonsContext.isTrainingContext()).thenReturn(true);
		
		Mockito.when(mockAxonsContext.isTrainingContext()).thenReturn(true);
		
		AxonsActivation leftToRightActivation = axons.pushLeftToRight(mockLeftToRightInputActivation, null, mockAxonsContext);
		Assert.assertNotNull(leftToRightActivation);
		if (expectPostDropoutInputToBeSet()) {
			NeuronsActivation postDropoutInput = leftToRightActivation.getPostDropoutInput().get();
			Assert.assertNotNull(postDropoutInput);
			Assert.assertEquals(getExpectedReformattedInputRows(), postDropoutInput.getFeatureCount());
			Assert.assertEquals(getExpectedReformattedInputColumns(), postDropoutInput.getExampleCount());
			Assert.assertEquals(mockLeftToRightInputActivation.getFeatureOrientation(), postDropoutInput.getFeatureOrientation());
		}

		NeuronsActivation postDropoutOutput = leftToRightActivation.getPostDropoutOutput();
		Assert.assertNotNull(postDropoutOutput);

		
	}
	
}
