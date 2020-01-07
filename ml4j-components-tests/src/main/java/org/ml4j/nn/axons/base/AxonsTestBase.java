package org.ml4j.nn.axons.base;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class AxonsTestBase<A extends Axons<?, ?, ?>> extends TestBase {

	private A axons;
	
	@Mock
	protected Neurons leftNeurons;
	
	@Mock
	protected Neurons rightNeurons;
	
	@Mock
	private AxonsContext mockAxonsContext;
	
	protected MatrixFactory matrixFactory;
	

	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		this.matrixFactory = createMatrixFactory();
		Mockito.when(leftNeurons.getNeuronCountExcludingBias()).thenReturn(100);
		Mockito.when(rightNeurons.getNeuronCountExcludingBias()).thenReturn(110);
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
		
		
		NeuronsActivation mockLeftToRightInputActivation = createNeuronsActivation(100, 32);
			
		AxonsActivation leftToRightActivation = axons.pushLeftToRight(mockLeftToRightInputActivation, null, mockAxonsContext);
		Assert.assertNotNull(leftToRightActivation);
		NeuronsActivation postDropoutInput = leftToRightActivation.getPostDropoutInput().get();
		Assert.assertNotNull(postDropoutInput);
		Assert.assertEquals(mockLeftToRightInputActivation.getFeatureCount(), postDropoutInput.getFeatureCount());
		Assert.assertEquals(mockLeftToRightInputActivation.getExampleCount(), postDropoutInput.getExampleCount());
		Assert.assertEquals(mockLeftToRightInputActivation.getFeatureOrientation(), postDropoutInput.getFeatureOrientation());

		NeuronsActivation postDropoutOutput = leftToRightActivation.getPostDropoutOutput();
		Assert.assertNotNull(postDropoutOutput);

		
	}
	
}
