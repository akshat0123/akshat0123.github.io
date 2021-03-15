import Sidebar from "../components/sidebar";
import Resume from "../content/resume.json";
import Header from "../components/header";
import Layout from "../components/layout";
import React from "react";


export default () => (
	<div>
        <Header/>
        <Layout>
            <h1>Professional Experience</h1>
            <ul className="nobullet">{ Resume.experience.map((data, index) => { return (
                <ul className="nobullet">
                    <li><b>{data.employer}</b></li>
                    <li><i>{data.position}</i></li>
                    <li>{data.dates}</li>
                    <li><ul>{data.description.map((desc, j) => {
                        return <li>{desc}</li>
                    })}</ul></li>
                    <br/>
                </ul>
            )})}
            </ul>
            <h1>Education</h1>
            <ul className="nobullet"> { Resume.education.map((data, index) => { return (<ul className="nobullet">
                <li><b>{data.university}</b></li>
                <li><i>{data.degree}</i></li>
                <li>{data.subject}</li>
                <li><b>{data.dates}</b></li>
                <br/>
            </ul>)})}</ul>
        </Layout>
        <Sidebar/>
	</div>
)
