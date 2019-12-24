package org.ml4j.nn.components.axons.base;

import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DirectedAxonsComponentTestBase {

	@Mock
	private NeuronsActivation mockNeuronsActivation;
	
	@Mock
	private AxonsContext mockAxonsContext;
	
	@Mock
	private Axons<Neurons, Neurons, ?> mockAxons;
	
	@Mock
	private DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	private MatrixFactory mockMatrixFactory;
	
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	}

	@SuppressWarnings("unchecked")
	private <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponent(L leftNeurons, R rightNeurons) {
		Mockito.when(mockAxons.getLeftNeurons()).thenReturn(leftNeurons);
		Mockito.when(mockAxons.getRightNeurons()).thenReturn(rightNeurons);
		return createDirectedAxonsComponentUnderTest((Axons<L, R, ?>)mockAxons);
	}
		
	protected abstract <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponentUnderTest(Axons<L, R, ?> axons);

	@Test
	public void testConstruction() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons, rightNeurons);
		Assert.assertNotNull(directedAxonsComponent);
	}
	
	@Test
	public void testGetComponentType() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons, rightNeurons);
		Assert.assertEquals(DirectedComponentType.AXONS, directedAxonsComponent.getComponentType());
	}
	
	@Test
	public void testDecompose() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons, rightNeurons);
		
		List<DefaultChainableDirectedComponent<?, ?>> components = directedAxonsComponent.decompose();
		Assert.assertNotNull(components);
		Assert.assertEquals(1, components.size());
		Assert.assertNotNull(components.get(0));
		Assert.assertEquals(directedAxonsComponent, components.get(0));
	}
	
	@Test
	public void testForwardPropagate() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons, rightNeurons);

		DirectedAxonsComponentActivation activation = directedAxonsComponent.forwardPropagate(mockNeuronsActivation, mockAxonsContext);
		Assert.assertNotNull(activation);
		
	}
	
	@Test
	public void testDup() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons, rightNeurons);

		DirectedAxonsComponent<?, ?, ?> dupComponent = directedAxonsComponent.dup();
		Assert.assertNotNull(dupComponent);
		Assert.assertNotEquals(directedAxonsComponent, dupComponent);
		
	}
	
	@Test
	public void testGetAxons() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons, rightNeurons);

		Axons<?, ?, ?> axons = directedAxonsComponent.getAxons();
		Assert.assertNotNull(axons);
	}
	
	@Test
	public void testGetContext() {
		
		
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons, rightNeurons);

		
		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(mockMatrixFactory);

		
		AxonsContext axonsContext = directedAxonsComponent.getContext(mockDirectedComponentsContext, 0);
		Assert.assertNotNull(axonsContext);
		Assert.assertNotSame(mockAxonsContext,axonsContext);

		Assert.assertEquals(0,  axonsContext.getRegularisationLambda(), 0.0000000001);
		Assert.assertEquals(1,  axonsContext.getLeftHandInputDropoutKeepProbability(), 0.0000000001);
		Assert.assertNotNull(axonsContext.getMatrixFactory());
		Assert.assertSame(mockMatrixFactory, axonsContext.getMatrixFactory());

	}

}
