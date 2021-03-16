import React, { Component } from "react";
import { Link } from "gatsby";

class Header extends Component {

	render() {
		return (
			<div id="header">
				<ul>
					<li><h3><Link to="/">Home</Link></h3></li>
					<li><h3><Link to="/experience/">Experience</Link></h3></li>
					<li><h3><Link to="/projects/">Projects</Link></h3></li>
					<li><h3><Link to="/blog/">Blog</Link></h3></li>
				</ul>
			</div>
		);
	}

}

export default Header
