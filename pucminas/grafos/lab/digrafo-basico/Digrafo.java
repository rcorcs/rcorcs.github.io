/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos.
 */

/**
 * Classe que implementa um grafo direcionado, ou digrafo.
 */
public abstract class Digrafo {

   /**
    * Adiciona uma nova aresta direcionada do vertice u para v.
    */
	public abstract void adicionaAresta(int u, int v);

   /**
    * Remove uma nova aresta direcionada do vertice u para v, caso a mesma exista.
    */
	public abstract void removeAresta(int u, int v);

   /**
    * Verifica se existe uma aresta direcionada do vertice u para v.
    * @return true caso existir a aresta (u,v) ou false caso contrario.
    */
	public abstract boolean existeAresta(int u, int v);

	/**
	 * Obtem o numero de vertices do digrafo.
    * @return o numero de vertices.
	 */
	public abstract int numVertices();

	/**
	 * Obtem o numero de arestas direcionadas do digrafo.
    * @return o numero de arestas do digrafo.
	 */
	public abstract int numArestas();

	/**
	 * Verifica se existe algum loop no Digrafo.
    * @return true caso existir algum loop em qualquer vertice ou false caso contrario.
	 */
	public abstract boolean existeLoop();

	/**
	 * Calcula o grau de saida do vertice u.
    * O grau de saida de um vertice u representa o numero de arestas direcionadas que possuem u como origem.
    * @return grau de saida do vertice u.
	 */
	public abstract int grauSaida(int u);

	/**
	 * Calcula o grau de entrada do vertice u.
    * O grau de entrada de um vertice u representa o numero de arestas direcionadas que possuem u como destino.
    * @return grau de entrada do vertice u.
	 */
	public abstract int grauEntrada(int u);

	/**
	 * Verifica se existe algum caminho com origem no vertice u e destino em v.
    * Os vertices u e v nao sao necessariamente adjacentes.
    * @return true caso existir algum de u para v ou false caso contrario.
	 */
	public abstract boolean existeCaminho(int u, int v);

}
