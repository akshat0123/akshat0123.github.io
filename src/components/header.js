import React, { Component } from "react";
import { Link } from "gatsby";

class Header extends Component {

	render() {
		return (
			<div id="header">
				<ul>
					<li><h3><Link to="/">Home</Link></h3></li>
					<li><h3><Link to="/cv/">CV</Link></h3></li>
					<li><h3><Link to="/teaching/">Teaching</Link></h3></li>
					<li><h3><Link to="/research/">Research</Link></h3></li>
				</ul>
			</div>
		);
	}

}

export default Header
